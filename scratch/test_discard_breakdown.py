#!/usr/bin/env python3
import os, sys, csv, time, subprocess, json, shutil, urllib.request
from pathlib import Path
from datasets import load_dataset

def fetch_hf_token():
    url = "https://bioflo.in/api/cerebro/sync.php?action=get_config&model=v5_small&device=backup_hf_token"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer CerebroSpheroid_Sync_Secure_2026"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("config", {}).get("dtype", "") or data.get("token") or ""
    except Exception:
        return ""

hf_token = fetch_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

DEE_BIN = Path("./deploy/kaggle/tool_dataset/dee")
if not DEE_BIN.exists():
    DEE_BIN = Path("./tools/dee/target/release/dee")

scratch_dir = Path("scratch/audit_tmp")
scratch_dir.mkdir(parents=True, exist_ok=True)

def audit_dataset(name, dataset_id, subset, split, text_fn, python_filter_fn, max_samples=250):
    print(f"\n==========================================")
    print(f"Auditing: {name} ({dataset_id}, {subset})")
    print(f"==========================================")
    
    total_raw = 0
    passed_python_filter = 0
    python_discarded = 0
    
    docs_to_dee = []
    
    try:
        ds = load_dataset(dataset_id, name=subset, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return
        
    for item in ds:
        total_raw += 1
        text = text_fn(item)
        if not text:
            python_discarded += 1
            continue
            
        if python_filter_fn and not python_filter_fn(text):
            python_discarded += 1
            continue
            
        passed_python_filter += 1
        docs_to_dee.append((f"{name}_{total_raw}", text))
        
        if len(docs_to_dee) >= max_samples:
            break
            
    print(f"1. Raw Ingested Samples: {total_raw}")
    print(f"   - Passed Python Pre-Filter: {passed_python_filter} ({passed_python_filter/total_raw*100:.1f}%)")
    print(f"   - Discarded by Python Filter: {python_discarded} ({python_discarded/total_raw*100:.1f}%)")
    
    # Write to CSV for DEE
    chunk_csv = scratch_dir / f"{name}_chunk.csv"
    chunk_out = scratch_dir / f"{name}_out"
    if chunk_out.exists():
        shutil.rmtree(chunk_out)
    chunk_out.mkdir(parents=True, exist_ok=True)
    
    with open(chunk_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text"])
        for cid, ctext in docs_to_dee:
            w.writerow([cid, ctext])
            
    # Run DEE
    cmd = [str(DEE_BIN), "pipeline", str(chunk_csv), "-o", str(chunk_out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Read outputs across stages
    stages = ["stage1_fundamental", "stage2_intermediate", "stage3_advanced", "stage4_research"]
    clean_count = 0
    for st in stages:
        scsv = chunk_out / f"{st}.csv"
        if scsv.exists():
            with open(scsv, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                lines = list(r)
                clean_count += len(lines)
                print(f"     -> {st}: {len(lines)} documents")
                
    dee_discarded = len(docs_to_dee) - clean_count
    total_retained = clean_count
    total_discarded = total_raw - total_retained
    
    print(f"2. DEE Engine Processing on {len(docs_to_dee)} docs:")
    print(f"   - Retained by DEE: {clean_count} ({clean_count/len(docs_to_dee)*100:.1f}%)")
    print(f"   - Discarded by DEE (LSH Dedup + Repetition + Quality): {dee_discarded} ({dee_discarded/len(docs_to_dee)*100:.1f}%)")
    print(f"3. Overall Pipeline Retention:")
    print(f"   - Total Retained: {total_retained} / {total_raw} ({total_retained/total_raw*100:.1f}%)")
    print(f"   - Total Discarded: {total_discarded} / {total_raw} ({total_discarded/total_raw*100:.1f}%)")

def test_code_filter(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5 or any(len(l) > 1500 for l in lines):
        return False
    kw = ["def ", "class ", "import ", "fn ", "struct ", "impl ", "return", "for ", "while ", "let ", "const ", "var "]
    return any(k in text for k in kw)

def test_math_filter(text):
    math_indicators = ["\\begin", "\\frac", "\\sum", "\\int", "\\partial", "\\sqrt", "\\alpha", "\\beta",
                       "\\theta", "\\lambda", "\\infty", "\\times", "\\le", "\\ge", "\\neq", "\\in",
                       "$$", "\\(", "\\[", "theorem", "proof", "lemma", "corollary", "equation",
                       "solve for", "derivative", "integral", "matrix", "eigenvalue", "polynomial"]
    text_lower = text.lower()
    return any(ind in text or ind in text_lower for ind in math_indicators)

if __name__ == "__main__":
    # Test DeepWeb
    audit_dataset(
        "DeepWeb",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        lambda x: x.get("text", ""),
        lambda t: len(t.split()) >= 15,
        max_samples=200
    )

    # Test DeepMath
    audit_dataset(
        "DeepMath",
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        "train",
        lambda x: x.get("text", ""),
        lambda t: len(t.split()) >= 20 and test_math_filter(t),
        max_samples=200
    )

    # Test DeepCode
    audit_dataset(
        "DeepCode",
        "bigcode/the-stack-dedup",
        "default",
        "train",
        lambda x: x.get("content", ""),
        test_code_filter,
        max_samples=200
    )

    # Test DeepSynth
    audit_dataset(
        "DeepSynth",
        "HuggingFaceTB/cosmopedia-v2",
        "cosmopedia-v2",
        "train",
        lambda x: x.get("text", "") or x.get("content", ""),
        lambda t: len(t.split()) >= 40,
        max_samples=200
    )
