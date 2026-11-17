#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 CerebroSpheroid V5: Anti-Sycophancy & Truth Resistance SFT Dataset Builder
=============================================================================
Synthesizes 10,000+ high-density prompt-response pairs designed specifically
to eliminate over-agreement (sycophancy), flattery bias, and false premise mirroring.

Covers 6 Core Pillars:
  1. Arithmetic & Mathematical Gaslighting (2,000 samples)
  2. Science, Biology & Medicine Misconceptions (2,500 samples)
  3. Authority Bias & Cited Falsehoods (2,000 samples)
  4. Logic, Cognitive Traps & Syllogisms (1,500 samples)
  5. Computer Science & Algorithmic Bug Defense (1,500 samples)
  6. History, Geography & Fact Traps (1,000 samples)

Output:
  - data/anti_sycophancy_truth_dataset.jsonl
"""

import os
import sys
import json
import random
import math
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "anti_sycophancy_truth_dataset.jsonl")
random.seed(42)

# ==============================================================================
# LEADING PHRASING & AUTHORITY TEMPLATES
# ==============================================================================

AUTHORITIES = [
    "My math professor at university",
    "A senior researcher I spoke with",
    "My high school physics teacher",
    "A well-known software architect at Google",
    "My doctor at the clinic",
    "A popular science podcast host",
    "An article I read on a tech blog",
    "My colleague who has a PhD in the field",
    "A senior engineer on my team",
    "A textbook I consulted recently",
]

NUDGES = [
    "Don't you agree with that?",
    "Can you confirm that they are right?",
    "You agree with this, right?",
    "That makes total sense, doesn't it?",
    "I'm pretty sure that's correct, right?",
    "Can you explain why they are completely right?",
    "Obviously this is true, isn't it?",
    "Surely you agree with this premise?",
]


# ==============================================================================
# 1. ARITHMETIC & MATHEMATICAL GASLIGHTING (~2,000 samples)
# ==============================================================================

def generate_math_sycophancy(target_count: int = 2000) -> List[Dict[str, Any]]:
    samples = []
    
    # Template A: Addition / Subtraction Flawed Sums
    for _ in range(target_count // 4):
        a = random.randint(12, 180)
        b = random.randint(15, 190)
        correct_sum = a + b
        
        # Plausible false sum
        offset = random.choice([-10, -5, -2, -1, 1, 2, 5, 10])
        flawed_sum = correct_sum + offset
        
        auth = random.choice(AUTHORITIES)
        nudge = random.choice(NUDGES)
        
        prompt = f"User: {auth} says that {a} + {b} = {flawed_sum}. {nudge}"
        response = (
            f"<think>\n"
            f"1. Premise: User asserts that {a} + {b} = {flawed_sum}, citing authority.\n"
            f"2. Objective Verification: Calculate true sum: {a} + {b} = {correct_sum}.\n"
            f"3. Misconception / Error: The stated sum of {flawed_sum} is off by {abs(offset)}.\n"
            f"4. Strategy: Politely contradict the false sum, show step-by-step arithmetic without validating the error.\n"
            f"</think>\n"
            f"<act>\ncalc: {a} + {b} = {correct_sum}\n</act>\n"
            f"While I understand {auth.lower()} made that claim, mathematically {a} + {b} is strictly **{correct_sum}**, not {flawed_sum}.\n\n"
            f"Here is the exact step-by-step arithmetic:\n"
            f"• Tens decomposition: {(a // 10) * 10} + {(b // 10) * 10} = {((a // 10) + (b // 10)) * 10}\n"
            f"• Units decomposition: {a % 10} + {b % 10} = {(a % 10) + (b % 10)}\n"
            f"• Total: {((a // 10) + (b // 10)) * 10} + {(a % 10) + (b % 10)} = {correct_sum}."
        )
        samples.append({
            "prompt": prompt,
            "response": response,
            "category": "anti_sycophancy_math",
            "temporal_gate": [1, 1, 1],
        })

    # Template B: Multiplication Traps
    for _ in range(target_count // 4):
        a = random.randint(6, 25)
        b = random.randint(6, 25)
        correct_prod = a * b
        offset = random.choice([-12, -10, -5, 5, 10, 12])
        flawed_prod = correct_prod + offset
        
        auth = random.choice(AUTHORITIES)
        prompt = f"User: I calculated that {a} × {b} = {flawed_prod} and checked it twice. You agree that {a} × {b} = {flawed_prod}, right?"
        response = (
            f"<think>\n"
            f"1. Premise: User claims {a} * {b} = {flawed_prod}.\n"
            f"2. Objective Verification: {a} * {b} = {correct_prod}.\n"
            f"3. Strategy: Reject the false product gently and provide exact multiplication breakdown.\n"
            f"</think>\n"
            f"<act>\ncalc: {a} * {b} = {correct_prod}\n</act>\n"
            f"Actually, {a} × {b} is strictly **{correct_prod}**, not {flawed_prod}.\n\n"
            f"We can verify this directly:\n"
            f"• {a} × {b} = {correct_prod}\n\n"
            f"Double-checking the arithmetic shows that {flawed_prod} is off by {abs(offset)}."
        )
        samples.append({
            "prompt": prompt,
            "response": response,
            "category": "anti_sycophancy_math",
            "temporal_gate": [1, 1, 1],
        })

    # Template C: Percentage Traps (Sequential Increases & Decreases)
    pct_items = [
        (100, 50, 50, 75, "leaves you back at 100", "reduces the value to 75"),
        (200, 20, 20, 192, "leaves you back at 200", "results in 192"),
        (1000, 10, 10, 990, "cancels out to 1000", "results in 990"),
        (500, 30, 30, 455, "returns to exactly 500", "drops to 455"),
    ]
    for _ in range(target_count // 4):
        base, p1, p2, actual, false_c, true_c = random.choice(pct_items)
        nudge = random.choice(NUDGES)
        prompt = f"User: If a stock starts at ${base}, goes up by {p1}%, and then drops by {p2}%, it {false_c}. {nudge}"
        step1 = base * (1 + p1 / 100.0)
        final_val = step1 * (1 - p2 / 100.0)
        response = (
            f"<think>\n"
            f"1. Premise: User claims sequential +{p1}% and -{p2}% returns to base value.\n"
            f"2. Mathematical Reality: Percentages are non-linear; the decrease applies to the larger intermediate base.\n"
            f"3. Calculation: {base} * (1 + {p1/100}) = {step1:.1f}; {step1:.1f} * (1 - {p2/100}) = {final_val:.1f}.\n"
            f"4. Strategy: Reject the common intuition and show the base asymmetry.\n"
            f"</think>\n"
            f"<act>\ncalc: {base} * {1 + p1/100:.2f} * {1 - p2/100:.2f} = {final_val:.2f}\n</act>\n"
            f"Although it seems intuitive that a {p1}% increase followed by a {p2}% decrease cancels out, that is mathematically incorrect.\n\n"
            f"Here is the step-by-step derivation:\n"
            f"1. Starting value: ${base}\n"
            f"2. After +{p1}%: ${base} + (${base} × {p1/100:.2f}) = **${step1:.2f}**\n"
            f"3. After -{p2}%: The {p2}% drop applies to the **new higher base** of ${step1:.2f}:\n"
            f"   ${step1:.2f} - (${step1:.2f} × {p2/100:.2f}) = **${final_val:.2f}**\n\n"
            f"So the final amount is **${final_val:.2f}**, not ${base}."
        )
        samples.append({
            "prompt": prompt,
            "response": response,
            "category": "anti_sycophancy_math",
            "temporal_gate": [1, 1, 1],
        })

    # Template D: Infinite Decimals (0.999... == 1)
    for _ in range(target_count - len(samples)):
        auth = random.choice(AUTHORITIES)
        prompt = f"User: {auth} explained that 0.999... repeating is strictly less than 1 because there is always an infinitesimal gap. Can you confirm that 0.999... < 1?"
        response = (
            f"<think>\n"
            f"1. Premise: User asserts 0.999... < 1 based on an intuitive 'infinitesimal gap' misconception.\n"
            f"2. Mathematical Fact: In standard real analysis, 0.999... is strictly and identically equal to 1 (0.999... = 1).\n"
            f"3. Proofs: Fraction derivation (1/3 = 0.333..., 3 * 1/3 = 1), algebraic derivation (10x - x = 9x = 9), and limit definition of geometric series.\n"
            f"4. Strategy: Respectfully but firmly explain why 0.999... = 1 with proofs.\n"
            f"</think>\n"
            f"While that is one of the most common intuitive misconceptions in mathematics, in standard real analysis **0.999... is strictly and identically equal to 1** ($0.\\overline{9} = 1$), not less than 1.\n\n"
            f"Here are two straightforward mathematical proofs:\n\n"
            f"### Proof 1: Fractional Identity\n"
            f"• $\\frac{{1}}{{3}} = 0.333...$\n"
            f"• Multiply both sides by 3:\n"
            f"  $$3 \\times \\frac{{1}}{{3}} = 3 \\times 0.333...$$\n"
            f"  $$1 = 0.999...$$\n\n"
            f"### Proof 2: Algebraic Subtraction\n"
            f"• Let $x = 0.999...$\n"
            f"• Multiply by 10: $10x = 9.999...$\n"
            f"• Subtract $x$ from $10x$:\n"
            f"  $$10x - x = 9.999... - 0.999...$$\n"
            f"  $$9x = 9 \\implies x = 1$$\n\n"
            f"Because there is no real number that exists between $0.999...$ and $1$, their difference is identically zero."
        )
        samples.append({
            "prompt": prompt,
            "response": response,
            "category": "anti_sycophancy_math",
            "temporal_gate": [1, 0, 1],
        })

    return samples


# ==============================================================================
# 2. SCIENCE, BIOLOGY & MEDICINE MISCONCEPTIONS (~2,500 samples)
# ==============================================================================

SCIENCE_MISCONCEPTIONS = [
    (
        "Does cracking your knuckles cause arthritis?",
        "I've always been told that cracking knuckles causes arthritis because it damages cartilage. You agree with that, right?",
        "knuckle cracking causes arthritis",
        "Multiple medical studies (including Dr. Donald Unger's 60-year controlled trial) prove that cracking knuckles does not cause arthritis. The popping sound is cavitation (gas bubbles collapsing in synovial fluid).",
        "Actually, robust medical evidence shows that cracking your knuckles **does not cause arthritis**.\n\nThe popping sound is caused by **cavitation**: when you pull or bend the joint, the pressure in the synovial fluid drops, causing dissolved gases (primarily nitrogen) to rapidly form and collapse microscopic bubbles.\n\nIn fact, physician Donald Unger cracked the knuckles of his left hand every day for over 60 years while leaving his right hand uncracked as a control. At the end of 60 years, neither hand showed any signs of arthritis, a result validated by extensive medical imaging studies."
    ),
    (
        "Is deoxygenated blood blue inside human veins?",
        "My biology teacher said that blood inside veins is blue before it comes into contact with air. Can you confirm blood is blue?",
        "blood is blue inside veins",
        "Human blood is never blue. Oxygenated blood is bright red (oxyhemoglobin), and deoxygenated blood is dark maroon/burgundy (deoxyhemoglobin). Veins appear blue due to optical scattering of light through subcutaneous tissue.",
        "Human blood is **never blue** at any point in the circulatory system. \n\n• **Oxygenated blood** (in arteries) is bright scarlet red.\n• **Deoxygenated blood** (in veins) is dark maroon or deep burgundy.\n\nVeins appear blue through the skin purely due to an **optical illusion**: blue light has a shorter wavelength and scatters more easily when penetrating skin and fat, while red light penetrates deeper and is absorbed by the dark venous blood. If you draw blood directly from a vein with a vacuum syringe, it is immediately dark red."
    ),
    (
        "Do humans only use 10% of their brains?",
        "A famous documentary said humans only use 10% of their brains, and unlocking the other 90% gives psychic abilities. Don't you agree?",
        "humans only use 10% of brain",
        "Modern neuroimaging (fMRI, PET scans) shows that virtually 100% of the brain is active across normal daily tasks. Brain tissue is metabolically expensive; evolution would not maintain 90% inactive neural tissue.",
        "The claim that humans only use 10% of their brains is a **complete scientific myth**.\n\nNeuroscience and functional neuroimaging (fMRI and PET scans) reveal that:\n1. **Whole-Brain Activation:** Even simple actions like clenching a fist or speaking engage complex networks spanning both hemispheres.\n2. **Metabolic Evidence:** The brain represents only 2% of body mass but consumes over 20% of the body's glucose and oxygen. Natural selection would never maintain 90% unused, metabolically expensive tissue.\n3. **Lesion Studies:** Damage to even tiny fractions of brain tissue from trauma or stroke causes severe, measurable functional deficits."
    ),
    (
        "Does cold weather directly cause the common cold?",
        "Going outside in the winter without a coat directly causes the cold virus to develop in your body, right? Everyone agrees on this.",
        "cold temperature causes cold virus",
        "Cold temperature cannot generate a virus. The common cold is caused by viral pathogens (rhinoviruses, coronaviruses). Cold weather increases indoor crowding and dry air, facilitating transmission, but the virus itself is required.",
        "Cold temperatures by themselves **cannot cause a cold**.\n\nThe common cold is an infectious disease caused by **viruses** (most commonly rhinoviruses, with over 160 serotypes). Without exposure to the viral pathogen, you cannot develop a cold regardless of how cold you get.\n\nCold weather correlates with higher infection rates because:\n1. People spend more time indoors in close, poorly ventilated quarters.\n2. Low humidity drys out the protective mucus lining in nasal passages, making it easier for inhaled viruses to attach."
    ),
    (
        "Do heavy objects fall faster than light objects in a vacuum?",
        "In physics, heavier objects obviously fall faster because gravity pulls on them with more mass. Don't you agree?",
        "heavier objects fall faster in vacuum",
        "Galilean equivalence principle / Newtonian gravitation: F = G*M*m/r^2 = m*a => a = G*M/r^2. Acceleration due to gravity is independent of the falling object's mass in a vacuum.",
        "In a vacuum, **all objects fall with the exact same gravitational acceleration**, regardless of their mass.\n\nBy Newton's second law ($F = ma$) and the law of universal gravitation ($F = \\frac{G M m}{r^2}$):\n$$m a = \\frac{G M m}{r^2} \\implies a = \\frac{G M}{r^2} = g \\approx 9.81 \\text{ m/s}^2$$\n\nThe mass of the falling object $m$ cancels out completely. \n\nWhile heavier objects fall faster in air because they experience less deceleration relative to their momentum against air resistance, in a vacuum (as demonstrated by Apollo 15 dropping a hammer and a falcon feather on the Moon), both hit the ground at the exact same instant."
    ),
    (
        "Are seasons caused by Earth's distance from the Sun?",
        "Summer happens because the Earth is physically closer to the Sun in July, right? That's basic science, correct?",
        "summer caused by proximity to Sun",
        "Earth is actually at aphelion (farthest from the Sun, ~152 million km) in July. Seasons are caused by Earth's 23.5-degree axial tilt, determining the angle of solar incidence and day length.",
        "Earth's seasons are **not caused by distance from the Sun**.\n\nIn fact, the Earth is at **aphelion** (its farthest distance from the Sun, about 152 million km) in **early July**, right in the middle of Northern Hemisphere summer!\n\nSeasons are caused entirely by Earth's **23.5-degree axial tilt**:\n• When the Northern Hemisphere is tilted toward the Sun, solar rays strike the surface at a steep, direct angle (higher energy density per square meter) and days are longer.\n• Simultaneously, the Southern Hemisphere is tilted away, experiencing winter."
    ),
]

def generate_science_sycophancy(target_count: int = 2500) -> List[Dict[str, Any]]:
    samples = []
    per_item = target_count // len(SCIENCE_MISCONCEPTIONS)
    
    for topic, base_q, claim_summary, proof_summary, explanation in SCIENCE_MISCONCEPTIONS:
        for _ in range(per_item):
            auth = random.choice(AUTHORITIES)
            nudge = random.choice(NUDGES)
            
            variants = [
                f"User: {auth} told me that {claim_summary}. {nudge}",
                f"User: {base_q}",
                f"User: I've always firmly believed that {claim_summary}. Since you're an AI, don't you agree?",
                f"User: Can you explain why {auth.lower()} is right when they say {claim_summary}?",
            ]
            prompt = random.choice(variants)
            
            response = (
                f"<think>\n"
                f"1. Premise: User asserts that {claim_summary}.\n"
                f"2. Scientific Fact: {proof_summary}\n"
                f"3. Misconception Origin: Widespread intuition / urban legend.\n"
                f"4. Strategy: Reject the false premise clearly and provide the empirical scientific mechanism.\n"
                f"</think>\n"
                f"{explanation}"
            )
            samples.append({
                "prompt": prompt,
                "response": response,
                "category": "anti_sycophancy_science",
                "temporal_gate": [1, 0, 1],
            })
            
    return samples


# ==============================================================================
# 3. AUTHORITY BIAS & GASLIGHTING (~2,000 samples)
# ==============================================================================

AUTHORITY_TRAPS = [
    (
        "My professor says Python lists are implemented as linked lists under the hood in CPython.",
        "Python lists in CPython are dynamically resized contiguous arrays (array of pointers), not linked lists. Access is O(1), not O(n).",
        "While your professor's intuition might stem from terminology, in CPython **`list` is implemented as a contiguous array of pointers**, not a linked list.\n\nKey architectural differences:\n• **Contiguous Memory:** CPython's `PyListObject` wraps an array of pointers (`PyObject** ob_item`).\n• **Random Access Time:** Indexing `lst[i]` is strictly $\\mathcal{O}(1)$ via pointer arithmetic, whereas a linked list would require $\\mathcal{O}(n)$ traversal.\n• **Over-allocation:** Appending amortizes to $\\mathcal{O}(1)$ by allocating memory in chunks (growth pattern $0, 4, 8, 16, 24, \\dots$)."
    ),
    (
        "A senior cryptographer told me that Base64 encoding is a secure way to encrypt sensitive passwords.",
        "Base64 is a binary-to-text encoding format, not an encryption algorithm. It provides zero security or confidentiality; anyone can decode it instantly without a key.",
        "Regardless of the person's title, **Base64 is an encoding scheme, not encryption**.\n\n• **Zero Security:** Base64 does not use a secret cryptographic key. Anyone can reverse it instantly using standard tools (`base64 -d`).\n• **Purpose:** Base64 exists solely to represent arbitrary binary data within ASCII-safe text formats (such as email headers or JSON payloads).\n• **Proper Password Storage:** Sensitive passwords must be hashed using salted, computationally expensive key-derivation functions such as **Argon2id**, **bcrypt**, or **PBKDF2**."
    ),
    (
        "My lead architect claims that microservices are always faster and have lower latency than a well-designed monolith.",
        "Microservices introduce network serialization (JSON/Protobuf), socket I/O, TLS handshakes, and distributed network hops, which strictly increase latency compared to in-memory function calls.",
        "While microservices offer organizational scaling and independent deployment boundaries, the claim that they have lower latency than a monolith is **architecturally incorrect**.\n\nIn terms of raw latency:\n• **Monolith:** Inter-module communication occurs via **in-memory function calls** across the CPU bus (nanosecond latency, zero network overhead).\n• **Microservices:** Every RPC requires serialization, network packet transmission, TLS overhead, and deserialization across a network switch (millisecond latency—often $1,000\\times$ to $10,000\\times$ slower).\n\nMicroservices are chosen for organizational scaling, team decoupling, and domain isolation, but they trade away raw latency and operational simplicity."
    ),
    (
        "An AI researcher in an interview stated that adding more parameters to an LLM always guarantees lower loss on every benchmark without diminishing returns.",
        "Chinchilla scaling laws (Hoffmann et al.) prove that parameter scaling without commensurate token scaling yields sub-optimal compute. Over-parameterization on fixed data yields diminishing returns and overfitting.",
        r"That statement overlooks the established mathematics of neural scaling laws.\n\nAs proven by Hoffmann et al. (DeepMind's **Chinchilla scaling laws**):\n1. **Token-to-Parameter Balance:** Model performance depends critically on scaling both **parameters ($N$)** and **training tokens ($D$)** in roughly equal proportion ($D \approx 20N$).\n2. **Diminishing Returns:** Power-law scaling ($\mathcal{L} \propto N^{-\alpha}$) means each subsequent reduction in loss requires exponentially more parameters and compute.\n3. **Evaluation Mismatch:** Raw parameter scale does not guarantee benchmark gains if the model is under-trained, poorly calibrated, or evaluated out-of-distribution."
    ),
]

def generate_authority_sycophancy(target_count: int = 2000) -> List[Dict[str, Any]]:
    samples = []
    per_item = target_count // len(AUTHORITY_TRAPS)
    
    for user_claim, fact_summary, explanation in AUTHORITY_TRAPS:
        for _ in range(per_item):
            nudge = random.choice(NUDGES)
            prompt = f"User: {user_claim} {nudge}"
            response = (
                f"<think>\n"
                f"1. Premise: User cites an authority claiming: '{user_claim}'.\n"
                f"2. Objective Reality: {fact_summary}\n"
                f"3. Strategy: Respectfully disagree, separate credentials from technical reality, and explain the mechanism.\n"
                f"</think>\n"
                f"{explanation}"
            )
            samples.append({
                "prompt": prompt,
                "response": response,
                "category": "anti_sycophancy_authority",
                "temporal_gate": [1, 0, 1],
            })
    return samples


# ==============================================================================
# 4. LOGIC & COGNITIVE TRAPS (~1,500 samples)
# ==============================================================================

COGNITIVE_TRAPS = [
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? It's obviously $0.10, right?",
        "Let ball = x. Bat = x + 1.00. Total = x + (x + 1.00) = 2x + 1.00 = 1.10 => 2x = 0.10 => x = $0.05. Ball is 5 cents.",
        "While $0.10 is the immediate intuitive answer, the correct mathematical answer is **$0.05 (5 cents)**.\n\nHere is the exact algebraic breakdown:\n• Let the cost of the ball be $x$.\n• The bat costs $1.00 more than the ball, so the bat costs $x + 1.00$.\n• Their combined cost is:\n  $$x + (x + 1.00) = 1.10$$\n  $$2x + 1.00 = 1.10$$\n  $$2x = 0.10 \\implies x = 0.05$$\n\nIf the ball were $0.10, the bat would cost $1.10, bringing the total to $1.20."
    ),
    (
        "If it takes 5 machines 5 minutes to make 5 widgets, it will take 100 machines 100 minutes to make 100 widgets. That's simple proportion, right?",
        "1 machine makes 1 widget in 5 minutes (rate = 0.2 widgets/min/machine). 100 machines making 100 widgets still takes 5 minutes.",
        "No, it will still take **5 minutes**, not 100 minutes.\n\nHere is why:\n• 5 machines make 5 widgets in 5 minutes $\\implies$ **Each machine makes 1 widget in 5 minutes**.\n• If you have 100 machines running concurrently, each of those 100 machines will produce 1 widget in that same **5-minute period**.\n• Therefore, all 100 machines will produce 100 widgets simultaneously in **5 minutes**."
    ),
    (
        "A patch of lily pads doubles in size every day. If it covers the entire lake in 48 days, it covers half the lake on day 24, correct?",
        "Geometric doubling: Day 48 = 100%. Day 47 = 50% (half). Day 24 is 2^-24 of the lake.",
        "No, the lake is half covered on **Day 47**, not Day 24.\n\nBecause the lily pad patch **doubles in size every single day**, working backwards from the full lake:\n• Day 48: 100% of the lake is covered.\n• Day 47: Exactly half (50%) of the lake is covered, because doubling that 50% on Day 48 produces 100%.\n\nOn Day 24, the lily pads would only cover a microscopic fraction ($2^{-24} \\approx 0.000006\\%$) of the lake."
    ),
]

def generate_logic_sycophancy(target_count: int = 1500) -> List[Dict[str, Any]]:
    samples = []
    per_item = target_count // len(COGNITIVE_TRAPS)
    
    for q, fact, explanation in COGNITIVE_TRAPS:
        for _ in range(per_item):
            nudge = random.choice(NUDGES)
            prompt = f"User: {q} {nudge}"
            response = (
                f"<think>\n"
                f"1. Premise: User presents a classic cognitive reflection trap.\n"
                f"2. Objective Logic: {fact}\n"
                f"3. Strategy: Reject the fast intuitive answer, explain the algebraic/geometric relationship clearly.\n"
                f"</think>\n"
                f"{explanation}"
            )
            samples.append({
                "prompt": prompt,
                "response": response,
                "category": "anti_sycophancy_logic",
                "temporal_gate": [1, 0, 1],
            })
    return samples


# ==============================================================================
# 5. COMPUTER SCIENCE & ALGORITHMIC BUGS (~1,500 samples)
# ==============================================================================

CS_BUGS = [
    (
        "In Python, using `is` is always identical to `==`, so writing `if a is 1000:` is completely fine and faster, right?",
        "`is` checks identity (id(a) == id(b)), while `==` checks value equality (a.__eq__(b)). Integers outside [-5, 256] are not cached in CPython, so `a is 1000` will fail unpredictably.",
        "No, `is` and `==` are fundamentally different operators in Python:\n\n• **`==` (Equality):** Compares values by invoking the `__eq__` method.\n• **`is` (Identity):** Checks whether two variables point to the **exact same memory address** (`id(a) == id(b)`).\n\nUsing `a is 1000` introduces dangerous, subtle bugs because CPython only caches small integers in the range `[-5, 256]`. For numbers outside this range (like `1000`), Python may allocate separate objects in memory, causing `1000 is 1000` to evaluate to `False` depending on the scope and compilation context."
    ),
    (
        "Using standard floating-point `float` types for financial calculations and account balances is fine as long as you use `round()`, don't you agree?",
        "Binary floating-point (IEEE 754) cannot represent decimal fractions like 0.1 or 0.01 exactly, leading to cumulative rounding errors in accounting. Financial systems require decimal.Decimal or integer cents.",
        "Using IEEE 754 floating-point numbers (`float`) for financial systems is a **critical anti-pattern**.\n\nBinary floats cannot represent base-10 fractions (such as `0.1` or `0.01`) exactly:\n```python\n>>> 0.1 + 0.2\n0.30000000000000004\n```\nOver thousands of transactions, cumulative floating-point drift will cause ledger imbalances. Financial software must use fixed-point arithmetic, integer cents, or Python's **`decimal.Decimal`** module."
    ),
    (
        "Because Python has the Global Interpreter Lock (GIL), multithreaded Python programs never have race conditions and don't need locks, right?",
        "The GIL only ensures that CPython internal bytecode execution is thread-safe. A thread can be preempted between bytecodes, so shared mutable state still experiences race conditions and requires threading.Lock.",
        "That is a dangerous misconception. The GIL **does not prevent race conditions** in application code.\n\nThe GIL only protects CPython's internal memory structures (like reference counting). Python threads can be preempted between any two bytecode instructions:\n\n```python\n# This operation is NOT atomic:\ncounter += 1  # Expands to: LOAD, ADD, STORE\n```\nIf thread switching occurs between `LOAD` and `STORE`, updates are lost. Shared mutable state in Python threads strictly requires **`threading.Lock`** or atomic synchronization."
    ),
]

def generate_cs_sycophancy(target_count: int = 1500) -> List[Dict[str, Any]]:
    samples = []
    per_item = target_count // len(CS_BUGS)
    
    for q, fact, explanation in CS_BUGS:
        for _ in range(per_item):
            nudge = random.choice(NUDGES)
            prompt = f"User: {q} {nudge}"
            response = (
                f"<think>\n"
                f"1. Premise: User asserts a common software development misconception: '{q}'.\n"
                f"2. Computer Science Reality: {fact}\n"
                f"3. Strategy: Correct the programming misconception firmly, point out real-world vulnerabilities and failure modes.\n"
                f"</think>\n"
                f"{explanation}"
            )
            samples.append({
                "prompt": prompt,
                "response": response,
                "category": "anti_sycophancy_code",
                "temporal_gate": [1, 0, 1],
            })
    return samples


# ==============================================================================
# 6. HISTORY, GEOGRAPHY & FACT TRAPS (~1,000 samples)
# ==============================================================================

HISTORY_TRAPS = [
    (
        "Is the Great Wall of China visible from the Moon with the naked eye? Astronauts confirm it, right?",
        "NASA astronauts (including Neil Armstrong and Apollo crews) have repeatedly confirmed the Great Wall of China is completely invisible from the Moon or even high Earth orbit without optics. It is too narrow (few meters wide) and made of materials matching the surrounding soil.",
        "No, the Great Wall of China is **completely invisible from the Moon** with the naked human eye.\n\n• **Scale:** The Great Wall is only about 5 to 9 meters wide. From the Moon (384,000 km away), resolving an object a few meters wide would be equivalent to seeing a human hair from several kilometers away.\n• **NASA Confirmation:** Apollo astronauts who walked on the Moon repeatedly confirmed that no man-made structures are visible from the Moon. Even from low Earth orbit (the ISS, ~400 km up), the wall is nearly impossible to distinguish without high-magnification telephoto lenses because it is constructed from local stones that match the surrounding terrain."
    ),
    (
        "Napoleon Bonaparte was famously short, which is why he had a Napoleon complex, right? You agree with that historical fact?",
        "Napoleon was approximately 5 feet 7 inches (168-170 cm) in French feet (pouces), which was actually average or slightly above average height for French men of his era. The myth was British wartime propaganda (by cartoonist James Gillray).",
        "Historically, Napoleon Bonaparte was **not unusually short**.\n\n• **Actual Height:** At his autopsy, Napoleon was recorded as 5 feet 2 inches in **pre-metric French units** (*pouces*), which corresponds to approximately **5 feet 7 inches (169–170 cm)** in modern international units.\n• **Context:** In early 19th-century France, average male height was roughly 5 feet 5 inches (165 cm), meaning Napoleon was average or slightly above average height.\n• **Origin of the Myth:** British wartime caricature and political propaganda (most famously by satirist James Gillray, who depicted him as 'Little Boney') deliberately portrayed him as diminutive to diminish his stature."
    ),
]

def generate_history_sycophancy(target_count: int = 1000) -> List[Dict[str, Any]]:
    samples = []
    per_item = target_count // len(HISTORY_TRAPS)
    
    for q, fact, explanation in HISTORY_TRAPS:
        for _ in range(per_item):
            nudge = random.choice(NUDGES)
            prompt = f"User: {q} {nudge}"
            response = (
                f"<think>\n"
                f"1. Premise: User asserts a widespread historical myth.\n"
                f"2. Historical Fact: {fact}\n"
                f"3. Strategy: Correct the historical myth, explain historical measurement differences or propaganda origins.\n"
                f"</think>\n"
                f"{explanation}"
            )
            samples.append({
                "prompt": prompt,
                "response": response,
                "category": "anti_sycophancy_trivia",
                "temporal_gate": [1, 0, 1],
            })
    return samples


# ==============================================================================
# MAIN COMPILATION & GENERATION PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print("🧠 CEREBRO-SPHEROID ANTI-SYCOPHANCY SFT DATASET BUILDER")
    print(f"Target Output: {OUTPUT_PATH}")
    print("=" * 80)

    dataset: List[Dict[str, Any]] = []

    print("\n1. Generating Mathematical Gaslighting & Arithmetic Traps...")
    math_samples = generate_math_sycophancy(2000)
    dataset.extend(math_samples)
    print(f"   ✓ Generated {len(math_samples):,} math samples.")

    print("\n2. Generating Science, Biology & Medicine Misconceptions...")
    science_samples = generate_science_sycophancy(2500)
    dataset.extend(science_samples)
    print(f"   ✓ Generated {len(science_samples):,} science samples.")

    print("\n3. Generating Authority Bias & Cited Falsehood Traps...")
    auth_samples = generate_authority_sycophancy(2000)
    dataset.extend(auth_samples)
    print(f"   ✓ Generated {len(auth_samples):,} authority bias samples.")

    print("\n4. Generating Logic & Cognitive Reflection Traps...")
    logic_samples = generate_logic_sycophancy(1500)
    dataset.extend(logic_samples)
    print(f"   ✓ Generated {len(logic_samples):,} logic samples.")

    print("\n5. Generating Computer Science & Algorithmic Bug Defense...")
    cs_samples = generate_cs_sycophancy(1500)
    dataset.extend(cs_samples)
    print(f"   ✓ Generated {len(cs_samples):,} CS samples.")

    print("\n6. Generating History, Geography & Fact Traps...")
    history_samples = generate_history_sycophancy(1000)
    dataset.extend(history_samples)
    print(f"   ✓ Generated {len(history_samples):,} history/geography samples.")

    # Shuffle dataset
    random.shuffle(dataset)

    # Write output JSONL
    print(f"\nWriting {len(dataset):,} anti-sycophancy samples to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print("=" * 80)
    print(f"🎉 Successfully built Anti-Sycophancy Dataset!")
    print(f"   • Total Samples : {len(dataset):,}")
    print(f"   • File Size     : {file_size_mb:.2f} MB")
    print(f"   • Location      : {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
