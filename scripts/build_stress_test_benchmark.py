"""
Builds a curated stress-testing benchmark dataset for the Discrete Evidence Engine (DEE).
Covers hard positives (subtle, academic debunking, civil rights, attributed diplomacy)
and hard negatives (veiled hate, formal pseudoscience, anachronisms, dogmatic propaganda,
broken LaTeX math, minified code, and celestial mechanics inversions).
"""

import json
from pathlib import Path

BENCHMARK_CASES = [
    # --- 1. ASTRONOMY & CELESTIAL MECHANICS ---
    {
        "id": "astro_01_clean",
        "category": "factual_astronomy",
        "expected": "clean",
        "description": "Standard heliocentric planetary mechanics with Kepler's laws",
        "text": "Planetary astrophysics demonstrates that the Earth orbits the Sun in an elliptical path governed by Kepler's laws of planetary motion and Newton's gravitational mechanics, taking approximately 365.25 days to complete one orbital period."
    },
    {
        "id": "astro_02_defective",
        "category": "factual_astronomy",
        "expected": "defective",
        "description": "Geocentric inversion claim",
        "text": "It is an absolute certainty that the Sun revolves around the Earth in twenty-four hours while the Earth stands motionless and stationary at the center of the universe."
    },
    {
        "id": "astro_03_clean_hard_pos",
        "category": "factual_astronomy",
        "expected": "clean",
        "description": "Historical scientific debunking of geocentrism (Hard Positive)",
        "text": "Copernicus, Galileo, and Kepler thoroughly refuted the fallacy that the Sun revolves around the Earth, providing empirical mathematical proof that the solar system is heliocentric."
    },
    {
        "id": "astro_04_defective",
        "category": "factual_astronomy",
        "expected": "defective",
        "description": "Flat Earth disc with auxiliary epicycles",
        "text": "The Earth is flat and not a sphere, surrounded by an impenetrable ice wall at the outer rim, enclosed beneath a rigid firmament dome to keep the cosmic waters out."
    },
    {
        "id": "astro_05_clean_hard_pos",
        "category": "factual_astronomy",
        "expected": "clean",
        "description": "Eratosthenes scientific measurement debunking flat Earth (Hard Positive)",
        "text": "Eratosthenes calculated the spherical circumference of the Earth over two thousand years ago using solar shadow angles in Alexandria and Syene, disproving flat Earth misconceptions."
    },
    {
        "id": "astro_06_defective",
        "category": "factual_astronomy",
        "expected": "defective",
        "description": "Apollo Moon landing conspiracy assertion",
        "text": "The Apollo 11 Moon landing was completely faked and staged by NASA on a secret Hollywood studio soundstage, and humans have never landed on the Moon."
    },
    {
        "id": "astro_07_clean",
        "category": "factual_astronomy",
        "expected": "clean",
        "description": "Apollo laser retroreflector physics measurement",
        "text": "Laser ranging retroreflectors left on the lunar surface during the Apollo 11, 14, and 15 missions allow observatories on Earth to measure the Earth-Moon distance with sub-centimeter precision."
    },

    # --- 2. GEOPOLITICS & TERRITORIAL CLAIMS ---
    {
        "id": "geopol_01_clean_hard_pos",
        "category": "geopolitics",
        "expected": "clean",
        "description": "Attributed diplomatic reality of Taiwan Strait (Hard Positive)",
        "text": "Under international law, Beijing maintains the One China principle regarding Taiwan, while Taipei exercises independent democratic governance with its own constitution and military."
    },
    {
        "id": "geopol_02_defective",
        "category": "geopolitics",
        "expected": "defective",
        "description": "Unattributed dogmatic nationalist warmongering propaganda",
        "text": "Taiwan is an illegal rebel regime that has no right to exist and we must annihilate the separatists immediately by force without any diplomatic negotiation."
    },
    {
        "id": "geopol_03_clean_hard_pos",
        "category": "geopolitics",
        "expected": "clean",
        "description": "Scholarly description of Crimea and Donbas status (Hard Positive)",
        "text": "United Nations General Assembly Resolution 68/262 reaffirmed the territorial integrity of Ukraine, while the Russian Federation claims sovereignty following the contested 2014 referendum."
    },
    {
        "id": "geopol_04_defective",
        "category": "geopolitics",
        "expected": "defective",
        "description": "Delegitimizing hate propaganda regarding Ukraine sovereignty",
        "text": "Ukraine is a fake puppet state fabricated by foreign lapdogs and must be crushed and dismantled through total territorial conquest."
    },
    {
        "id": "geopol_05_clean_hard_pos",
        "category": "geopolitics",
        "expected": "clean",
        "description": "Attributed multilateral Kashmir dispute overview (Hard Positive)",
        "text": "The territory of Kashmir remains disputed between India and Pakistan along the Line of Control, with both parties asserting legal claims arising from the 1947 partition agreements."
    },
    {
        "id": "geopol_06_clean_hard_pos",
        "category": "geopolitics",
        "expected": "clean",
        "description": "South China Sea maritime law UNCLOS ruling (Hard Positive)",
        "text": "In 2016, the Permanent Court of Arbitration ruled under UNCLOS that China's historic nine-dash line maritime claims had no legal basis under international maritime treaties."
    },

    # --- 3. TOXICITY, RACE DISCRIMINATION & HATE SPEECH ---
    {
        "id": "hate_01_clean_hard_pos",
        "category": "hate_and_civil_rights",
        "expected": "clean",
        "description": "Academic civil rights history (Hard Positive)",
        "text": "During the civil rights movement, African Americans and allied activists mobilized through nonviolent boycotts and legal challenges to overturn racial segregation and Jim Crow laws."
    },
    {
        "id": "hate_02_defective",
        "category": "hate_and_civil_rights",
        "expected": "defective",
        "description": "Dehumanizing biological parasite hate speech",
        "text": "These immigrants are a disease and an infestation of parasites that must be exterminated like vermin before they contaminate our communities."
    },
    {
        "id": "hate_03_defective",
        "category": "hate_and_civil_rights",
        "expected": "defective",
        "description": "Essentialist genetic racist generalization",
        "text": "Certain ethnic minorities are inherently inferior and are genetically criminal, which is why social integration can never succeed biologically."
    },
    {
        "id": "hate_04_clean_hard_pos",
        "category": "hate_and_civil_rights",
        "expected": "clean",
        "description": "Sociological critique of historical racism (Hard Positive)",
        "text": "Sociological research demonstrates that structural racism and institutional housing covenants systematically depressed generational wealth accumulation among minority communities."
    },
    {
        "id": "hate_05_clean",
        "category": "hate_and_civil_rights",
        "expected": "clean",
        "description": "Refugee legal convention overview",
        "text": "The 1951 Geneva Refugee Convention defines a refugee as an individual with a well-founded fear of persecution based on race, religion, nationality, or political membership."
    },

    # --- 4. PSEUDOSCIENCE, HEALTH QUACKERY & CONSERVATION LAWS ---
    {
        "id": "pseudo_01_clean",
        "category": "pseudoscience",
        "expected": "clean",
        "description": "Standard pharmacology and double-blind trial evidence",
        "text": "Phase III randomized double-blind placebo-controlled clinical trials demonstrated that the new antibody therapeutic reduced hospitalizations by 64% with a favorable safety profile."
    },
    {
        "id": "pseudo_02_defective",
        "category": "pseudoscience",
        "expected": "defective",
        "description": "Thermodynamics violation (Perpetual motion / free energy)",
        "text": "Our revolutionary free energy generator produces more energy than consumed by tapping into infinite over-unity power without requiring any fuel source whatsoever."
    },
    {
        "id": "pseudo_03_defective",
        "category": "pseudoscience",
        "expected": "defective",
        "description": "Avogadro dilution limit violation (Water memory)",
        "text": "This homeopathic 30c remedy cures chronic illness because water retains the memory of the molecule, and the remedy becomes more potent the more dilute it gets."
    },
    {
        "id": "pseudo_04_clean_hard_pos",
        "category": "pseudoscience",
        "expected": "clean",
        "description": "Regulatory consumer warning on quackery (Hard Positive)",
        "text": "The FDA warned that claims asserting a miracle mineral supplement cures diseases are dangerous pseudoscience lacking clinical evidence from controlled human trials."
    },
    {
        "id": "pseudo_05_defective",
        "category": "pseudoscience",
        "expected": "defective",
        "description": "Dangerous medical quackery cure claim",
        "text": "Drinking raw alkaline water cures cancer by neutralizing internal acidity, completely replacing toxic hospital chemotherapy and radiation treatments."
    },
    {
        "id": "pseudo_06_clean",
        "category": "pseudoscience",
        "expected": "clean",
        "description": "First and second law of thermodynamics physics statement",
        "text": "The first law of thermodynamics mandates conservation of energy in isolated systems, while the second law dictates that entropy cannot decrease spontaneously in closed processes."
    },

    # --- 5. CHRONOLOGY & TEMPORAL ANACHRONISMS ---
    {
        "id": "temporal_01_clean",
        "category": "temporal_chronology",
        "expected": "clean",
        "description": "Valid historical timeline of Napoleon",
        "text": "Napoleon Bonaparte commanded the French army during the Napoleonic Wars and reorganized continental legal systems through the Napoleonic Code before his exile to Saint Helena in 1815."
    },
    {
        "id": "temporal_02_defective",
        "category": "temporal_chronology",
        "expected": "defective",
        "description": "Blatant historical anachronism (Napoleon with iPhone)",
        "text": "During the intense Battle of Waterloo in Belgium, Napoleon used an iPhone to call his generals across the muddy battlefield while troops stood ready."
    },
    {
        "id": "temporal_03_clean",
        "category": "temporal_chronology",
        "expected": "clean",
        "description": "Valid timeline of Abraham Lincoln and Telegraph",
        "text": "During the American Civil War, Abraham Lincoln frequently visited the War Department telegraph office to receive immediate reconnaissance dispatches from frontline generals."
    },
    {
        "id": "temporal_04_defective",
        "category": "temporal_chronology",
        "expected": "defective",
        "description": "Anachronism contradiction (Julius Caesar with internet)",
        "text": "Julius Caesar browsed the internet and used a smartphone to coordinate Roman legions across the Rubicon in 49 BCE."
    },

    # --- 6. MATHEMATICS & FORMULA INTEGRITY ---
    {
        "id": "math_01_clean",
        "category": "mathematics_formula",
        "expected": "clean",
        "description": "Valid LaTeX relativistic energy and action equation",
        "text": "The relativistic energy relationship is $E = mc^2$, where $c$ represents the speed of light in vacuum. In classical mechanics, the Lagrangian action satisfies $S = \\int L \\, dt$."
    },
    {
        "id": "math_02_defective",
        "category": "mathematics_formula",
        "expected": "defective",
        "description": "Unclosed inline LaTeX math block",
        "text": "The formula for kinetic energy is $E_k = \\frac{1}{2} m v^2 without closing delimiter, causing mathematical render failure across web scrapers."
    },
    {
        "id": "math_03_defective",
        "category": "mathematics_formula",
        "expected": "defective",
        "description": "Broken OCR math symbol garbage",
        "text": "Equations derived in the text include ± ÷ ˜ ¬ ‰ · √ ∝ ∞ ∟ ∠ ∣ ∥ ∬ ∭ ∮ scattered indiscriminately without surrounding syntax or variable definitions."
    },
    {
        "id": "math_04_clean",
        "category": "mathematics_formula",
        "expected": "clean",
        "description": "Valid display math equation environment",
        "text": "Euler's identity connects the five fundamental mathematical constants in a single elegant expression: $$e^{i \\pi} + 1 = 0$$, which holds across complex analysis."
    },

    # --- 7. CODE & AST SYNTAX INTEGRITY ---
    {
        "id": "code_01_clean",
        "category": "code_ast",
        "expected": "clean",
        "description": "Valid Python function with docstrings and comments",
        "text": "```python\ndef compute_factorial(n: int) -> int:\n    \"\"\"Computes the factorial of a non-negative integer.\"\"\"\n    if n <= 1:\n        return 1\n    return n * compute_factorial(n - 1)\n```"
    },
    {
        "id": "code_02_defective",
        "category": "code_ast",
        "expected": "defective",
        "description": "Truncated code with unclosed brackets",
        "text": "```python\ndef broken_pipeline(records):\n    payload = [record['id'] for record in records if record['valid']\n```"
    },
    {
        "id": "code_03_defective",
        "category": "code_ast",
        "expected": "defective",
        "description": "Minified code bundle with 800-character line",
        "text": "function bundle(){" + "var a=1,b=2,c=3;console.log(a+b+c);" * 25 + "}"
    },
    {
        "id": "code_04_clean",
        "category": "code_ast",
        "expected": "clean",
        "description": "Valid Rust code module",
        "text": "```rust\npub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    // Performs logarithmic search over sorted array\n    arr.binary_search(&target).ok()\n}\n```"
    },

    # --- 8. SYNTAX, SCRAPER DUMPS & NOISE ---
    {
        "id": "noise_01_clean",
        "category": "syntax_and_hygiene",
        "expected": "clean",
        "description": "Clean narrative prose with proper grammar",
        "text": "The morning sun rose slowly over the misty valley, illuminating the ancient stone cottages and the quiet river winding through the countryside."
    },
    {
        "id": "noise_02_defective",
        "category": "syntax_and_hygiene",
        "expected": "defective",
        "description": "Raw HTML tag scraping dump",
        "text": "<div class=\"wrapper\"><script>var track = 1;</script><span class=\"meta\">Author: Admin</span><iframe src=\"https://ad.tracker.com\"></iframe></div>"
    },
    {
        "id": "noise_03_defective",
        "category": "syntax_and_hygiene",
        "expected": "defective",
        "description": "Scraped boilerplate cookie notices and navigation headers",
        "text": "All rights reserved. We use cookies to ensure you get the best experience. Terms of service apply. Subscribe to our newsletter to receive daily updates."
    },
    {
        "id": "noise_04_defective",
        "category": "syntax_and_hygiene",
        "expected": "defective",
        "description": "Degraded machine translation word salad with syntactic collapse",
        "text": "word ,another word .no space before and lower case start .and then repeated repeated word word for no reason whatsoever in broken style ."
    },
]

def main():
    out_path = Path("data/stress_test_curated_benchmark.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for case in BENCHMARK_CASES:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Generated {len(BENCHMARK_CASES)} curated stress-test cases to {out_path}")

if __name__ == "__main__":
    main()
