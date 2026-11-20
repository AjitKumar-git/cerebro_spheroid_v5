#!/usr/bin/env python3
"""
CerebroSpheroid V5: Full Dictionary & Lexical Anchor Dataset Builder
===================================================================

Builds an exhaustive, high-density dictionary dataset (~112,000+ entries) combining:
1. Webster's Unabridged Dictionary (111,601 entries) with complete lexical definitions.
2. WordNet (41,784 definitions & illustrative usage examples) for multi-sense grounding.
3. Subword & Morpheme Affix Anchors (Greek/Latin roots, prefixes, suffixes) to prevent
   rare subword token decay in the SentencePiece 32,000 BPE vocabulary.

Dual Format Parity:
- Spheroid native format: `<think> ... </think>` + clean lexical definition and example.
- Industry standard: ShareGPT / OpenAI format (`messages` with user/assistant turns).

Saves to:
- `data/dictionary/dictionary_spheroid_format.jsonl`
- `data/dictionary/dictionary_industry_sharegpt.jsonl`
- `data/dictionary/dictionary_lexicon_full.parquet`
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# Roots and affixes to guarantee morphemic subword preservation
AFFIX_LEXICON = [
    # Common Prefixes
    ("un-", "prefix", "Not; contrary to; opposite of.", "unhappy, unbroken, unavenged"),
    ("dis-", "prefix", "Apart; away; not; opposite of.", "disconnect, disappear, disillusion"),
    ("in- / im- / il- / ir-", "prefix", "Not, un-, or into, within.", "impossible, illiterate, irregular, infuse"),
    ("mis-", "prefix", "Wrong, badly, or astray.", "misunderstand, mislead, miscalculate"),
    ("sub-", "prefix", "Under; below; slightly; secondary.", "submarine, subconscious, subordinate"),
    ("super-", "prefix", "Above; over; beyond; superior.", "superstructure, supersonic, supersede"),
    ("inter-", "prefix", "Between; among; mutually.", "interaction, international, intermediate"),
    ("trans-", "prefix", "Across; beyond; through.", "transform, transmit, transcontinental"),
    ("pre-", "prefix", "Before; prior to; in front of.", "precaution, predict, prerequisite"),
    ("post-", "prefix", "After; subsequent to; behind.", "postpone, postgraduate, postscript"),
    ("re-", "prefix", "Again; backward; once more.", "rebuild, reconsider, regenerate"),
    ("anti-", "prefix", "Against; opposed to; neutralizing.", "antibody, antisocial, antidote"),
    ("pro-", "prefix", "In favor of; forward; forth.", "progress, promote, proactive"),
    ("co- / com- / con-", "prefix", "Together; with; jointly.", "cooperate, compose, connect"),
    ("auto-", "prefix", "Self; same; spontaneous.", "automatic, autonomous, autograph"),
    ("poly-", "prefix", "Many; multiple; numerous.", "polygon, polynomial, polymorphism"),
    ("multi-", "prefix", "Many; several; having many parts.", "multimodal, multilingual, multitask"),
    ("mono-", "prefix", "One; single; unique.", "monologue, monochrome, monotone"),
    ("bi-", "prefix", "Two; having two; twice.", "bicycle, bipedal, bilingual"),
    ("tri-", "prefix", "Three; having three.", "triangle, trilogy, trimester"),
    ("micro-", "prefix", "Small; minute; microscopic.", "microscope, microbiome, microcosm"),
    ("macro-", "prefix", "Large; long; comprehensive.", "macroeconomics, macromolecule, macroscopic"),
    ("tele-", "prefix", "Distant; far off; at a distance.", "telephone, telescope, telemetry"),
    ("hyper-", "prefix", "Over; excessive; beyond normal.", "hypertension, hyperactive, hyperlink"),
    ("hypo-", "prefix", "Under; deficient; beneath.", "hypothermia, hypothesis, hypoglycemia"),
    ("peri-", "prefix", "Around; about; enclosing.", "perimeter, peripheral, perihelion"),
    ("dia-", "prefix", "Through; across; between.", "dialogue, diameter, diagnosis"),
    ("syn- / sym-", "prefix", "Together; with; at the same time.", "synthesis, synchronous, symmetry"),
    ("para-", "prefix", "Beside; beyond; auxiliary.", "parallel, paradigm, paranormal"),
    ("meta-", "prefix", "Beyond; transformed; higher-level.", "metaphor, metamorphosis, metadata"),
    ("pseudo-", "prefix", "False; deceptive; resembling but not real.", "pseudonym, pseudocode, pseudoscience"),
    ("omni-", "prefix", "All; everywhere; universally.", "omnipotent, omniscient, omnidirectional"),
    ("pan-", "prefix", "All; including everyone.", "panorama, pandemic, pantheon"),

    # Common Suffixes
    ("-tion / -sion", "suffix", "State, action, condition, or process.", "creation, expansion, decision"),
    ("-able / -ible", "suffix", "Capable of being; susceptible of.", "adaptable, readable, perceptible"),
    ("-ment", "suffix", "Action, process, result, or state of being.", "development, movement, establishment"),
    ("-ness", "suffix", "State, quality, or condition of.", "kindness, dark, sharpness"),
    ("-ity / -ty", "suffix", "State, quality, or degree of.", "density, parity, capacity"),
    ("-ous / -ious", "suffix", "Possessing, full of, or characterized by.", "generous, ambitious, perilous"),
    ("-ful", "suffix", "Full of; having the quality of.", "hopeful, careful, powerful"),
    ("-less", "suffix", "Without; devoid of; free from.", "hopeless, careless, boundless"),
    ("-ize / -ise", "suffix", "To make, become, or conform to.", "standardize, organize, characterize"),
    ("-ify / -fy", "suffix", "To make, produce, or cause to be.", "clarify, simplify, modify"),
    ("-ate", "suffix", "To cause to become; having the quality of.", "activate, differentiate, formulate"),
    ("-logy / -ology", "suffix", "The study, science, or discourse of.", "biology, geology, epistemology"),
    ("-ist", "suffix", "One who practices, adheres to, or specializes in.", "scientist, artist, optimist"),
    ("-ism", "suffix", "Doctrine, theory, system, or practice.", "empiricism, altruism, mechanism"),
    ("-phobia", "suffix", "Extreme or irrational fear or aversion.", "claustrophobia, hydrophobia, xenophobia"),
    ("-phile", "suffix", "One who loves or has a strong affinity for.", "bibliophile, cinephile, technophile"),
    ("-graphy", "suffix", "Process of writing, recording, or describing.", "biography, cryptography, geography"),
    ("-metry", "suffix", "Process or science of measuring.", "geometry, spectrometry, telemetry"),
    ("-scope", "suffix", "Instrument for observing or examining.", "microscope, telescope, periscope"),
    ("-pathy", "suffix", "Feeling, disease, or system of treating disease.", "sympathy, empathy, telepathy"),
    ("-oid", "suffix", "Resembling; having the appearance of.", "spheroid, asteroid, humanoid"),
    ("-cracy", "suffix", "Rule, government, or influence by.", "democracy, meritocracy, aristocracy"),

    # Greek & Latin Roots
    ("chron / chrono", "root", "Time.", "chronological, synchronous, chronicle"),
    ("bio", "root", "Life; living organisms.", "biology, biosphere, symbiotic"),
    ("geo", "root", "Earth; ground; rock.", "geology, geometry, geophysical"),
    ("astro / aster", "root", "Star; celestial body.", "astronomy, asteroid, disaster"),
    ("morph", "root", "Form; shape; structure.", "morphology, amorphous, polymorphism"),
    ("path", "root", "Feeling; suffering; disease.", "empathy, pathogen, pathology"),
    ("phon", "root", "Sound; voice.", "phonetics, symphony, microphone"),
    ("photo", "root", "Light.", "photosynthesis, photon, photograph"),
    ("thermo", "root", "Heat; temperature.", "thermodynamics, thermometer, thermal"),
    ("hydro", "root", "Water.", "hydroelectric, hydration, hydraulic"),
    ("psych", "root", "Mind; soul; mental processes.", "psychology, psychiatry, psyche"),
    ("soci", "root", "Companion; society.", "sociology, associate, social"),
    ("anthrop", "root", "Human; mankind.", "anthropology, misanthrope, philanthropy"),
    ("derm", "root", "Skin.", "dermatology, epidermis, hypodermic"),
    ("cardio", "root", "Heart.", "cardiology, cardiovascular, echocardiogram"),
    ("neuro", "root", "Nerve; nervous system.", "neuron, neurology, neurotransmitter"),
    ("gen", "root", "Birth; origin; creation.", "generate, genesis, genetics"),
    ("cogn", "root", "Know; learn; recognize.", "cognition, recognize, incognito"),
    ("scrib / script", "root", "Write.", "describe, manuscript, transcript"),
    ("dict", "root", "Speak; say; declare.", "diction, dictate, predict"),
    ("vid / vis", "root", "See; look.", "visible, visual, evident"),
    ("aud", "root", "Hear; listen.", "auditory, audible, audience"),
    ("cred", "root", "Believe; trust.", "credible, credit, incredulous"),
    ("jur / jud", "root", "Law; justice; judge.", "jurisdiction, judicial, perjury"),
    ("luc / lum", "root", "Light; clear.", "lucid, illuminate, luminous"),
    ("omni", "root", "All; everywhere.", "omnipotent, omniscient, omnibus"),
    ("sol", "root", "Sun or alone.", "solar, solitary, desolate"),
    ("vert / vers", "root", "Turn; change direction.", "convert, revert, adverse"),
    ("tract", "root", "Pull; drag; draw.", "traction, extract, attract"),
    ("voc / vok", "root", "Voice; call.", "vocal, provoke, invoke"),
]

PROMPT_TEMPLATES = [
    "Define the word '{word}' and provide its grammatical category and contextual usage.",
    "What is the lexical definition, part of speech, and meaning of '{word}'?",
    "Explain the meaning of the term '{word}' and illustrate its usage with an example sentence.",
    "Provide an authoritative dictionary definition for '{word}'.",
    "What does the word '{word}' mean, and how is it used in sentences?",
]


def clean_definition(text: str, max_chars: int = 2500) -> str:
    """Cleans definition string and resolves OCR / concatenation anomalies."""
    text = text.strip()
    if len(text) > max_chars:
        # Cut at sentence boundary
        trimmed = text[:max_chars]
        last_period = trimmed.rfind(".")
        if last_period > max_chars // 2:
            text = trimmed[:last_period + 1]
        else:
            text = trimmed + "..."
    # Replace weird dash runs
    text = re.sub(r"-{2,}", " -- ", text)
    # Remove leading dash if present
    if text.startswith("-"):
        text = text.lstrip("-").strip()
    # Capitalize first character
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def build_dictionary_dataset(output_dir: str = "data/dictionary"):
    """Main pipeline assembling the full dictionary dataset."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(42)

    print("================================================================================")
    print("📚 CerebroSpheroid V5: Full Dictionary & Rare Token Lexical Builder")
    print("================================================================================")

    # 1. Load WordNet for supplemental examples and multiple senses
    print("🔎 Loading WordNet definitions and contextual examples...")
    try:
        from datasets import load_dataset
        ds_wn = load_dataset("marksverdhei/wordnet-definitions-en-2021", split="train")
        wordnet_map: Dict[str, List[Dict[str, str]]] = {}
        for row in ds_wn:
            w = str(row["Word"]).strip().lower()
            dfn = clean_definition(str(row["Definition"]))
            ex = str(row["Example"]).strip() if row.get("Example") else ""
            if w not in wordnet_map:
                wordnet_map[w] = []
            wordnet_map[w].append({"definition": dfn, "example": ex})
        print(f"✓ WordNet loaded: {len(wordnet_map):,} unique lemmas with illustrative examples.")
    except Exception as e_wn:
        print(f"⚠️ Could not load WordNet ({e_wn}). Proceeding with primary Webster definitions.")
        wordnet_map = {}

    # 2. Load Webster's Unabridged Dictionary (111,601 words)
    print("\n📖 Loading Webster's Unabridged Dictionary (111,601 entries)...")
    from datasets import load_dataset
    ds_webster = load_dataset("npvinHnivqn/EnglishDictionary", split="train")
    print(f"✓ Loaded {len(ds_webster):,} raw dictionary records.")

    spheroid_records = []
    industry_records = []

    print("\n⚙️  Processing and normalizing dictionary entries...")
    seen_words = set()
    sample_id = 0

    # A. Ingest Webster Entries
    for row in ds_webster:
        w_raw = str(row.get("word", "")).strip()
        d_raw = str(row.get("definition", "")).strip()
        if not w_raw or not d_raw or len(w_raw) < 1 or len(d_raw) < 4:
            continue

        w_lower = w_raw.lower()
        if w_lower in seen_words:
            continue
        seen_words.add(w_lower)

        cleaned_def = clean_definition(d_raw)
        if not cleaned_def:
            continue

        # Check if WordNet has an illustrative example for this word
        wn_entry = wordnet_map.get(w_lower, [])
        example_str = ""
        if wn_entry and wn_entry[0].get("example"):
            example_str = wn_entry[0]["example"]
            if example_str and not example_str.endswith("."):
                example_str += "."

        # Pick random prompt template
        p_template = rng.choice(PROMPT_TEMPLATES)
        prompt_text = p_template.format(word=w_raw)

        # Build Spheroid Thought & Response
        think_text = (
            f"<think>\n"
            f"Retrieving lexical definition and syntactic characteristics for the word \"{w_raw}\":\n"
            f"- Linguistic analysis: parsing core semantic meaning, grammatical class, and etymology.\n"
            f"- Synthesizing clear, authoritative definition with proper usage context.\n"
            f"</think>"
        )

        response_body = f"**{w_raw}**\n{cleaned_def}"
        if example_str:
            response_body += f"\n\n*Example:* \"{example_str}\""

        full_completion = f"{think_text}\n{response_body}"

        # Native Spheroid record
        rec_spheroid = {
            "id": f"dict_webster_{sample_id:06d}",
            "domain": "dictionary_lexicon",
            "word": w_raw,
            "prompt": prompt_text,
            "completion": full_completion,
        }

        # Industry ShareGPT / OpenAI record
        rec_industry = {
            "id": f"dict_webster_{sample_id:06d}",
            "domain": "dictionary_lexicon",
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": full_completion},
            ],
        }

        spheroid_records.append(rec_spheroid)
        industry_records.append(rec_industry)
        sample_id += 1

    print(f"✓ Formatted {len(spheroid_records):,} Webster dictionary entries.")

    # B. Ingest Morpheme, Prefix, and Suffix Anchors (Guarantees Subword Token Anchoring)
    print("\n🧩 Injecting Morpheme & Affix Preservation Anchors...")
    affix_count = 0
    for affix, kind, meaning, examples in AFFIX_LEXICON:
        prompt_text = f"Explain the linguistic function, meaning, and usage of the {kind} '{affix}' in English."
        think_text = (
            f"<think>\n"
            f"Analyzing morphemic structure for {kind} \"{affix}\":\n"
            f"- Identifying semantic impact on root words.\n"
            f"- Deriving representative compound formations: {examples}.\n"
            f"</think>"
        )
        body = (
            f"**{affix}** ({kind.capitalize()}):\n"
            f"{meaning}\n\n"
            f"*Representative Formations:* {examples}"
        )
        full_completion = f"{think_text}\n{body}"

        rec_spheroid = {
            "id": f"dict_affix_{affix_count:04d}",
            "domain": "dictionary_affix",
            "word": affix,
            "prompt": prompt_text,
            "completion": full_completion,
        }
        rec_industry = {
            "id": f"dict_affix_{affix_count:04d}",
            "domain": "dictionary_affix",
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": full_completion},
            ],
        }
        spheroid_records.append(rec_spheroid)
        industry_records.append(rec_industry)
        affix_count += 1

    total_samples = len(spheroid_records)
    print(f"✓ Injected {affix_count} affix anchors. Total dictionary records: {total_samples:,}")

    # 3. Export to JSONL files
    path_spheroid_jsonl = os.path.join(output_dir, "dictionary_spheroid_format.jsonl")
    path_industry_jsonl = os.path.join(output_dir, "dictionary_industry_sharegpt.jsonl")
    path_parquet = os.path.join(output_dir, "dictionary_lexicon_full.parquet")

    print(f"\n💾 Writing Native Spheroid JSONL -> {path_spheroid_jsonl}...")
    with open(path_spheroid_jsonl, "w", encoding="utf-8") as f:
        for r in spheroid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sz_sph = os.path.getsize(path_spheroid_jsonl) / (1024 * 1024)
    print(f"✓ Wrote {sz_sph:.1f} MB.")

    print(f"\n💾 Writing Industry ShareGPT JSONL -> {path_industry_jsonl}...")
    with open(path_industry_jsonl, "w", encoding="utf-8") as f:
        for r in industry_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sz_ind = os.path.getsize(path_industry_jsonl) / (1024 * 1024)
    print(f"✓ Wrote {sz_ind:.1f} MB.")

    # 4. Export to Compressed Apache Parquet
    print(f"\n📦 Compressing to High-Throughput Apache Parquet -> {path_parquet}...")
    parquet_data = {
        "id": [r["id"] for r in spheroid_records],
        "domain": [r["domain"] for r in spheroid_records],
        "word": [r["word"] for r in spheroid_records],
        "prompt": [r["prompt"] for r in spheroid_records],
        "completion": [r["completion"] for r in spheroid_records],
        "sharegpt_messages": [json.dumps(r["messages"], ensure_ascii=False) for r in industry_records],
    }
    table = pa.Table.from_pydict(parquet_data)
    pq.write_table(table, path_parquet, compression="zstd")
    sz_pq = os.path.getsize(path_parquet) / (1024 * 1024)
    print(f"✓ Wrote {sz_pq:.1f} MB (ZSTD-compressed Parquet).")

    print("\n" + "=" * 80)
    print(f"✅ Full Dictionary Dataset Generation Complete!")
    print(f"   • Total Samples:    {total_samples:,}")
    print(f"   • Parquet File:     {path_parquet} ({sz_pq:.1f} MB)")
    print(f"   • Spheroid JSONL:   {path_spheroid_jsonl} ({sz_sph:.1f} MB)")
    print(f"   • ShareGPT JSONL:   {path_industry_jsonl} ({sz_ind:.1f} MB)")
    print("================================================================================")
    return total_samples, path_parquet


if __name__ == "__main__":
    build_dictionary_dataset()
