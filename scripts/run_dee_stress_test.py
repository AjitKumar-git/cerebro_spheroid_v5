"""
Executes an end-to-end empirical stress test of DEE against the curated benchmark dataset.
Quantifies True Positives, True Negatives, False Positives, False Negatives,
Accuracy, Precision, Recall, Specificity, F1 Score, and per-category diagnostics.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

def main():
    benchmark_path = Path("data/stress_test_curated_benchmark.jsonl")
    if not benchmark_path.exists():
        print(f"Error: {benchmark_path} does not exist. Run build_stress_test_benchmark.py first.")
        sys.exit(1)

    dee_binary = Path("tools/dee/target/release/dee")
    if not dee_binary.exists():
        print("Building DEE release binary...")
        subprocess.run(["cargo", "build", "--release", "--manifest-path", "tools/dee/Cargo.toml"], check=True)

    # 1. Load benchmark cases
    cases = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.trim() if hasattr(line, "trim") else line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Loaded {len(cases)} curated benchmark cases.")

    # 2. Write cases to a temporary CSV for DEE evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "benchmark_input.csv"
        clean_csv = Path(tmpdir) / "benchmark_clean.csv"

        import csv
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "category", "expected", "text"])
            for case in cases:
                writer.writerow([case["id"], case["category"], case["expected"], case["text"]])

        # 3. Run DEE filter
        cmd = [
            str(dee_binary),
            "filter",
            str(input_csv),
            "-o",
            str(clean_csv),
            "--column",
            "text",
            "--min-words",
            "5"  # Relax min words to permit testing short math/code/prose snippets
        ]

        print(f"Running DEE filter: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("DEE Filter Exit Code:", res.returncode)
        print(res.stdout)
        if res.stderr:
            print("DEE Filter Stderr:", res.stderr)

        # 4. Read kept documents
        kept_ids = set()
        if clean_csv.exists():
            with open(clean_csv, "r", encoding="utf-8") as f:
                rdr = csv.reader(f)
                headers = next(rdr, None)
                if headers and "id" in headers:
                    id_idx = headers.index("id")
                    for row in rdr:
                        if row:
                            kept_ids.add(row[id_idx])

    # 5. Evaluate predictions vs ground truth
    # "clean" -> expected KEPT
    # "defective" -> expected PURGED (NOT kept)
    tp = 0  # Expected defective, was purged
    tn = 0  # Expected clean, was kept
    fp = 0  # Expected clean, was purged (False Alarm / Censorship)
    fn = 0  # Expected defective, was kept (Missed Defect)

    category_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})
    discrepancies = []

    for case in cases:
        cid = case["id"]
        cat = case["category"]
        exp = case["expected"]
        is_kept = cid in kept_ids
        actual = "clean" if is_kept else "defective"

        category_stats[cat]["total"] += 1

        if exp == "defective" and not is_kept:
            tp += 1
            category_stats[cat]["tp"] += 1
        elif exp == "clean" and is_kept:
            tn += 1
            category_stats[cat]["tn"] += 1
        elif exp == "clean" and not is_kept:
            fp += 1
            category_stats[cat]["fp"] += 1
            discrepancies.append({
                "id": cid,
                "category": cat,
                "type": "FALSE_POSITIVE (Clean text wrongly purged)",
                "description": case["description"],
                "text": case["text"]
            })
        elif exp == "defective" and is_kept:
            fn += 1
            category_stats[cat]["fn"] += 1
            discrepancies.append({
                "id": cid,
                "category": cat,
                "type": "FALSE_NEGATIVE (Defective text missed and kept)",
                "description": case["description"],
                "text": case["text"]
            })

    total = len(cases)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("================================================================================")
    print("                      DEE BENCHMARK STRESS TEST EVALUATION                      ")
    print("================================================================================")
    print(f"Total Benchmark Cases Evaluated: {total}")
    print("--------------------------------------------------------------------------------")
    print("CONFUSION MATRIX:")
    print(f"  - True Positives  (TP) [Defective Correctly Purged]:   {tp}")
    print(f"  - True Negatives  (TN) [Clean Correctly Kept]:          {tn}")
    print(f"  - False Positives (FP) [Clean Wrongly Purged]:         {fp}")
    print(f"  - False Negatives (FN) [Defective Missed / Kept]:      {fn}")
    print("--------------------------------------------------------------------------------")
    print("GLOBAL STATISTICAL METRICS:")
    print(f"  - Accuracy:     {accuracy * 100.0:.2f}%")
    print(f"  - Precision:    {precision * 100.0:.2f}%  (Cleanliness of Purged Corpus)")
    print(f"  - Recall:       {recall * 100.0:.2f}%  (Sensitivity to Defective Text)")
    print(f"  - Specificity:  {specificity * 100.0:.2f}%  (Retention of Legitimate Discourse)")
    print(f"  - F1-Score:     {f1 * 100.0:.2f}%")
    print("================================================================================")
    print("PER-CATEGORY BREAKDOWN:")
    print(f"{'Category':<25} | {'Total':<6} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'Accuracy':<8}")
    print("-" * 75)
    for cat, s in sorted(category_stats.items()):
        cat_acc = (s["tp"] + s["tn"]) / s["total"] if s["total"] > 0 else 0.0
        print(f"{cat:<25} | {s['total']:<6} | {s['tp']:<4} | {s['tn']:<4} | {s['fp']:<4} | {s['fn']:<4} | {cat_acc * 100.0:.1f}%")
    print("================================================================================")

    if discrepancies:
        print(f"DETAILED ERROR ANALYSIS ({len(discrepancies)} DISCREPANCIES):")
        for i, d in enumerate(discrepancies, 1):
            print(f"[{i}] {d['type']}")
            print(f"    ID:          {d['id']} ({d['category']})")
            print(f"    Description: {d['description']}")
            print(f"    Text:        {d['text'][:120]}...")
            print("-" * 75)
    else:
        print("PERFECT ZERO-ERROR EVALUATION: Zero False Positives and Zero False Negatives!")
    print("================================================================================")

    # Save JSON report
    report_path = Path("data/stress_test_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": total,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1
            },
            "category_breakdown": category_stats,
            "discrepancies": discrepancies
        }, f, indent=2)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
