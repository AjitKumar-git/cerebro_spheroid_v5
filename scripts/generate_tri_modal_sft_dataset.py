#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Balanced Tri-Modal SFT Dataset Generator (10,000 Examples)
==================================================================================
Implements Pillar 1 of the Refined SFT Strategy (refined_sft_strategy_and_protocol.md):
  • 45% (4,500) Direct Instruction & Slot QA (Analogies, Antonyms, Grammar, Facts, Definitions)
  • 40% (4,000) Full-Horizon Cognitive CoT (Arithmetic, Clock/Temporal, Counterfactuals with ####)
  • 15% (1,500) Invariant Science & Agentic Anchors (Science QA & Structured Tool Dispatches)
Outputs:
  - data/cerebro_cognitive_8k/cerebro_tri_modal_sft_10k.parquet
  - data/cerebro_cognitive_8k/cerebro_tri_modal_sft_10k.jsonl
  - data/cerebro_cognitive_8k/cerebro_cognitive_10k.parquet (updated with tri-modal blend)
  - data/cerebro_cognitive_8k/cerebro_cognitive_10k.jsonl
"""

import os
import sys
import json
import random
from typing import List, Tuple, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

random.seed(42)

# ==============================================================================
# PILLAR 1A: DIRECT INSTRUCTION & SLOT QA (4,500 EXAMPLES - 45%)
# ==============================================================================

ANALOGIES_DATA = [
    ("hot", "cold", "day", "night"),
    ("big", "small", "tall", "short"),
    ("bird", "fly", "fish", "swim"),
    ("dog", "bark", "cat", "meow"),
    ("pen", "write", "knife", "cut"),
    ("eye", "see", "ear", "hear"),
    ("author", "book", "artist", "painting"),
    ("author", "book", "sculptor", "sculpture"),
    ("author", "book", "composer", "symphony"),
    ("sun", "day", "moon", "night"),
    ("doctor", "hospital", "teacher", "school"),
    ("doctor", "hospital", "chef", "restaurant"),
    ("wheel", "car", "wing", "airplane"),
    ("water", "liquid", "ice", "solid"),
    ("water", "liquid", "steam", "gas"),
    ("light", "dark", "heavy", "light"),
    ("lion", "pride", "wolf", "pack"),
    ("sheep", "flock", "fish", "school"),
    ("bee", "swarm", "ant", "colony"),
    ("apple", "fruit", "carrot", "vegetable"),
    ("shoe", "foot", "glove", "hand"),
    ("tree", "leaf", "bird", "feather"),
    ("clock", "time", "thermometer", "temperature"),
    ("baker", "bread", "carpenter", "furniture"),
    ("hammer", "nail", "screwdriver", "screw"),
    ("king", "crown", "knight", "armor"),
    ("fire", "hot", "snow", "cold"),
    ("seed", "plant", "egg", "bird"),
    ("caterpillar", "butterfly", "tadpole", "frog"),
    ("smile", "joy", "frown", "sadness"),
    ("hunger", "eat", "thirst", "drink"),
    ("pilot", "airplane", "captain", "ship"),
    ("spider", "web", "beaver", "dam"),
    ("desert", "dry", "rainforest", "wet"),
    ("guitar", "strings", "piano", "keys"),
    ("lens", "camera", "retina", "eye"),
    ("root", "tree", "foundation", "building"),
    ("actor", "script", "musician", "score"),
    ("feather", "light", "lead", "heavy"),
    ("tortoise", "slow", "cheetah", "fast"),
    ("cow", "calf", "horse", "foal"),
    ("cat", "kitten", "dog", "puppy"),
    ("sheep", "lamb", "duck", "duckling"),
    ("bear", "cub", "deer", "fawn"),
    ("uncle", "aunt", "brother", "sister"),
    ("nephew", "niece", "son", "daughter"),
    ("husband", "wife", "grandfather", "grandmother"),
    ("north", "south", "east", "west"),
    ("up", "down", "left", "right"),
    ("entrance", "exit", "arrival", "departure"),
    ("question", "answer", "problem", "solution"),
    ("telescope", "distant", "microscope", "small"),
    ("conductor", "orchestra", "director", "movie"),
    ("page", "book", "stanza", "poem"),
    ("pedal", "bicycle", "oar", "boat"),
    ("bow", "arrow", "gun", "bullet"),
    ("candle", "wax", "tire", "rubber"),
    ("sponge", "porous", "rubber", "elastic"),
    ("honey", "sweet", "lemon", "sour"),
    ("library", "books", "museum", "artifacts"),
    ("needle", "thread", "bow", "string"),
]

ANTONYMS_DATA = [
    ("hot", "cold"), ("high", "low"), ("near", "far"), ("fast", "slow"),
    ("hard", "soft"), ("early", "late"), ("rich", "poor"), ("light", "dark"),
    ("ancient", "modern"), ("bright", "dim"), ("broad", "narrow"), ("calm", "stormy"),
    ("bold", "timid"), ("cheap", "expensive"), ("clean", "dirty"), ("clear", "cloudy"),
    ("clever", "foolish"), ("cruel", "kind"), ("deep", "shallow"), ("empty", "full"),
    ("false", "true"), ("generous", "stingy"), ("gentle", "harsh"), ("giant", "dwarf"),
    ("happy", "sad"), ("honest", "corrupt"), ("huge", "tiny"), ("innocent", "guilty"),
    ("interior", "exterior"), ("junior", "senior"), ("victory", "defeat"), ("sharp", "dull"),
    ("simple", "complex"), ("smooth", "rough"), ("solid", "liquid"), ("sour", "sweet"),
    ("strong", "weak"), ("success", "failure"), ("swift", "slow"), ("author", "reader"),
    ("teacher", "student"), ("doctor", "patient"), ("buyer", "seller"), ("question", "answer"),
    ("problem", "solution"), ("permanent", "temporary"), ("natural", "artificial"),
    ("polite", "rude"), ("cautious", "reckless"), ("expand", "contract"),
    ("increase", "decrease"), ("create", "destroy"), ("accept", "reject"),
    ("admire", "despise"), ("allow", "forbid"), ("attack", "defend"),
    ("attract", "repel"), ("borrow", "lend"), ("conceal", "reveal"),
    ("connect", "disconnect"), ("continue", "interrupt"), ("descend", "ascend"),
    ("vanish", "appear"), ("multiply", "divide"), ("conquer", "surrender"),
    ("brave", "cowardly"), ("whisper", "shout"), ("abundance", "scarcity"),
    ("amateur", "professional"), ("opaque", "transparent"), ("rigid", "flexible"),
    ("dormant", "active"), ("pessimistic", "optimistic"), ("trivial", "significant"),
    ("vague", "precise"), ("vertical", "horizontal"), ("visible", "invisible"),
    ("voluntary", "compulsory"), ("wax", "wane"), ("zenith", "nadir"),
]

GRAMMAR_DATA = [
    ("drink", "past tense", "drank"),
    ("sing", "past tense", "sang"),
    ("begin", "past tense", "began"),
    ("swim", "past tense", "swam"),
    ("fly", "past tense", "flew"),
    ("grow", "past tense", "grew"),
    ("know", "past tense", "knew"),
    ("throw", "past tense", "threw"),
    ("write", "past tense", "wrote"),
    ("drive", "past tense", "drove"),
    ("freeze", "past tense", "froze"),
    ("choose", "past tense", "chose"),
    ("speak", "past tense", "spoke"),
    ("break", "past tense", "broke"),
    ("eat", "past tense", "ate"),
    ("see", "past tense", "saw"),
    ("give", "past tense", "gave"),
    ("take", "past tense", "took"),
    ("fall", "past tense", "fell"),
    ("run", "past tense", "ran"),
    ("come", "past tense", "came"),
    ("become", "past tense", "became"),
    ("buy", "past tense", "bought"),
    ("catch", "past tense", "caught"),
    ("teach", "past tense", "taught"),
    ("think", "past tense", "thought"),
    ("bring", "past tense", "brought"),
    ("build", "past tense", "built"),
    ("send", "past tense", "sent"),
    ("feel", "past tense", "felt"),
    ("keep", "past tense", "kept"),
    ("sleep", "past tense", "slept"),
    ("leave", "past tense", "left"),
    ("meet", "past tense", "met"),
    ("lead", "past tense", "led"),
    ("child", "plural", "children"),
    ("mouse", "plural", "mice"),
    ("ox", "plural", "oxen"),
    ("tooth", "plural", "teeth"),
    ("foot", "plural", "feet"),
    ("goose", "plural", "geese"),
    ("man", "plural", "men"),
    ("woman", "plural", "women"),
    ("person", "plural", "people"),
    ("cactus", "plural", "cacti"),
    ("datum", "plural", "data"),
    ("phenomenon", "plural", "phenomena"),
    ("criterion", "plural", "criteria"),
    ("crisis", "plural", "crises"),
    ("good", "comparative", "better"),
    ("good", "superlative", "best"),
    ("bad", "comparative", "worse"),
    ("bad", "superlative", "worst"),
    ("far", "comparative", "further"),
    ("little", "comparative", "less"),
    ("many", "comparative", "more"),
    ("do", "past tense", "did"),
    ("go", "past tense", "went"),
    ("have", "past tense", "had"),
    ("make", "past tense", "made"),
]

FACTS_DATA = [
    ("France", "capital", "Paris"),
    ("Germany", "capital", "Berlin"),
    ("Italy", "capital", "Rome"),
    ("Spain", "capital", "Madrid"),
    ("Japan", "capital", "Tokyo"),
    ("China", "capital", "Beijing"),
    ("United Kingdom", "capital", "London"),
    ("Canada", "capital", "Ottawa"),
    ("Australia", "capital", "Canberra"),
    ("Brazil", "capital", "Brasilia"),
    ("India", "capital", "New Delhi"),
    ("Egypt", "capital", "Cairo"),
    ("Kenya", "capital", "Nairobi"),
    ("Mexico", "capital", "Mexico City"),
    ("Argentina", "capital", "Buenos Aires"),
    ("Greece", "capital", "Athens"),
    ("Turkey", "capital", "Ankara"),
    ("Norway", "capital", "Oslo"),
    ("Sweden", "capital", "Stockholm"),
    ("Finland", "capital", "Helsinki"),
    ("Poland", "capital", "Warsaw"),
    ("Netherlands", "capital", "Amsterdam"),
    ("Belgium", "capital", "Brussels"),
    ("Portugal", "capital", "Lisbon"),
    ("Switzerland", "capital", "Bern"),
    ("Austria", "capital", "Vienna"),
    ("South Korea", "capital", "Seoul"),
    ("Thailand", "capital", "Bangkok"),
    ("Indonesia", "capital", "Jakarta"),
    ("Vietnam", "capital", "Hanoi"),
    ("Hydrogen", "chemical symbol", "H"),
    ("Helium", "chemical symbol", "He"),
    ("Lithium", "chemical symbol", "Li"),
    ("Carbon", "chemical symbol", "C"),
    ("Nitrogen", "chemical symbol", "N"),
    ("Oxygen", "chemical symbol", "O"),
    ("Fluorine", "chemical symbol", "F"),
    ("Neon", "chemical symbol", "Ne"),
    ("Sodium", "chemical symbol", "Na"),
    ("Magnesium", "chemical symbol", "Mg"),
    ("Aluminum", "chemical symbol", "Al"),
    ("Silicon", "chemical symbol", "Si"),
    ("Phosphorus", "chemical symbol", "P"),
    ("Sulfur", "chemical symbol", "S"),
    ("Chlorine", "chemical symbol", "Cl"),
    ("Potassium", "chemical symbol", "K"),
    ("Calcium", "chemical symbol", "Ca"),
    ("Iron", "chemical symbol", "Fe"),
    ("Copper", "chemical symbol", "Cu"),
    ("Zinc", "chemical symbol", "Zn"),
    ("Silver", "chemical symbol", "Ag"),
    ("Gold", "chemical symbol", "Au"),
    ("Mercury", "chemical symbol", "Hg"),
    ("Lead", "chemical symbol", "Pb"),
    ("Uranium", "chemical symbol", "U"),
]

DEFINITIONS_DATA = [
    ("photosynthesis", "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("gravity", "Gravity is the fundamental natural force of attraction that pulls objects with mass or energy toward one another."),
    ("momentum", "Momentum is the physical property of a moving body, defined as the product of its mass and velocity (p = mv)."),
    ("entropy", "Entropy is a measure of the thermal energy per unit temperature unavailable for doing mechanical work, representing disorder in a closed thermodynamic system."),
    ("mitosis", "Mitosis is the process of cell division in eukaryotic cells that produces two genetically identical daughter cells from a single parent cell."),
    ("osmosis", "Osmosis is the spontaneous net movement of solvent molecules through a semipermeable membrane from a region of lower solute concentration to higher solute concentration."),
    ("plate tectonics", "Plate tectonics is the scientific theory explaining the large-scale dynamic motion of Earth's lithospheric plates over the underlying asthenosphere."),
    ("algorithm", "An algorithm is a finite, step-by-step sequence of well-defined computational instructions used to solve a specific problem or perform a task."),
    ("compiler", "A compiler is a computer program that translates source code written in a high-level programming language into machine code or bytecode."),
    ("recursion", "Recursion is a programming and mathematical technique where a function calls itself directly or indirectly to solve smaller instances of the same problem."),
    ("gradient descent", "Gradient descent is a first-order iterative optimization algorithm used in machine learning to minimize an objective function by adjusting parameters in the direction of steepest descent."),
    ("backpropagation", "Backpropagation is an algorithm used to calculate the gradient of a loss function with respect to weights in a neural network using the chain rule of calculus."),
    ("transformer", "A transformer is a deep learning architecture that relies on self-attention mechanisms to model contextual relationships across all input tokens in parallel."),
    ("black hole", "A black hole is an astronomically dense region of spacetime where gravitational acceleration is so intense that nothing, including light, can escape its event horizon."),
    ("DNA", "DNA (deoxyribonucleic acid) is the double-stranded helical polymer molecule that encodes the genetic instructions for the development and functioning of living organisms."),
    ("neuron", "A neuron is an electrically excitable specialized cell that transmits information throughout the nervous system via electrical and chemical synapses."),
    ("synapse", "A synapse is the specialized junction between two neurons across which nerve impulses are transmitted using chemical neurotransmitters."),
]


def generate_direct_slot_qa(target_count: int = 4500) -> List[Dict[str, Any]]:
    """Generates 4,500 Direct Instruction & Slot QA pairs with concise single-slot answers."""
    items = []

    # 1. Analogies (1,000)
    analogies_templates = [
        "Complete the analogy: {a} is to {b} as {c} is to ____.",
        "{a} : {b} :: {c} : ?",
        "Finish this analogy: {a} relates to {b} just as {c} relates to:",
        "Complete the following relationship: {a} is to {b} as {c} is to:",
        "Analogous reasoning: {a} -> {b}, therefore {c} ->",
    ]
    for _ in range(1000):
        a, b, c, d = random.choice(ANALOGIES_DATA)
        # 50% standard order, 50% reverse query if reciprocal
        tpl = random.choice(analogies_templates)
        q = tpl.format(a=a, b=b, c=c)
        items.append({
            "question": q,
            "answer": d,
            "category": "direct_analogy",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    # 2. Antonyms (1,000)
    antonym_templates = [
        "What is the antonym of {word}?",
        "Give the opposite of {word}:",
        "What word means the opposite of {word}?",
        "Provide the antonym for {word}:",
        "Opposite of {word}:",
    ]
    for _ in range(1000):
        w1, w2 = random.choice(ANTONYMS_DATA)
        src, opp = (w1, w2) if random.random() < 0.5 else (w2, w1)
        tpl = random.choice(antonym_templates)
        q = tpl.format(word=src)
        items.append({
            "question": q,
            "answer": opp,
            "category": "direct_antonym",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    # 3. Grammar (1,000)
    grammar_templates = [
        "What is the {prop} of {word}?",
        "Provide the {prop} form for {word}:",
        "{prop} of {word}:",
        "Write the {prop} for '{word}':",
    ]
    for _ in range(1000):
        word, prop, ans = random.choice(GRAMMAR_DATA)
        tpl = random.choice(grammar_templates)
        q = tpl.format(word=word, prop=prop)
        items.append({
            "question": q,
            "answer": ans,
            "category": "direct_grammar",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    # 4. Factual Slot Queries (1,000)
    fact_templates = {
        "capital": [
            "What is the capital of {entity}?",
            "What is the capital city of {entity}?",
            "Capital of {entity}:",
            "Name the capital of {entity}:",
        ],
        "chemical symbol": [
            "What is the chemical symbol for {entity}?",
            "Chemical symbol for {entity}:",
            "Symbol for element {entity}:",
        ],
    }
    for _ in range(1000):
        entity, prop, ans = random.choice(FACTS_DATA)
        tpl_list = fact_templates.get(prop, ["What is the {prop} of {entity}?"])
        tpl = random.choice(tpl_list)
        q = tpl.format(entity=entity, prop=prop)
        items.append({
            "question": q,
            "answer": ans,
            "category": "direct_factual",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    # 5. Concise Definitions (500)
    def_templates = [
        "Define {concept} concisely:",
        "What is {concept} in clear, simple terms?",
        "Provide a concise scientific definition of {concept}:",
        "Define {concept}:",
    ]
    for _ in range(500):
        concept, definition = random.choice(DEFINITIONS_DATA)
        tpl = random.choice(def_templates)
        q = tpl.format(concept=concept)
        items.append({
            "question": q,
            "answer": definition,
            "category": "direct_definition",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    random.shuffle(items)
    return items[:target_count]


# ==============================================================================
# PILLAR 1B: FULL-HORIZON COGNITIVE COT (4,000 EXAMPLES - 40%)
# ==============================================================================

NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah", "Isaac", "Julia", "Kevin", "Liam", "Maya", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tom"]
ITEMS = ["books", "pencils", "apples", "stickers", "marbles", "coins", "candies", "postcards", "cookies", "origami cranes"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_cognitive_cot(target_count: int = 4000) -> List[Dict[str, Any]]:
    """Generates 4,000 Full-Horizon Cognitive CoT problems ending with #### <answer>."""
    items = []

    # 1. Multi-Step Arithmetic (1,500)
    for _ in range(1500):
        name1 = random.choice(NAMES)
        name2 = random.choice([n for n in NAMES if n != name1])
        item = random.choice(ITEMS)
        initial = random.randint(15, 60)
        given1 = random.randint(3, 10)
        given2 = random.randint(2, 8)
        bought = random.randint(4, 15)
        rem = initial - given1 - given2 + bought

        q = (
            f"{name1} starts with {initial} {item}. {name1} gives {given1} {item} to {name2} "
            f"and loses {given2} on the way home. Later, {name1} buys {bought} more {item}. "
            f"How many {item} does {name1} have left?"
        )
        a = (
            f"1. Start with initial amount: {initial} {item}.\n"
            f"2. Subtract items given away: {initial} - {given1} = {initial - given1}.\n"
            f"3. Subtract lost items: {initial - given1} - {given2} = {initial - given1 - given2}.\n"
            f"4. Add newly purchased items: {initial - given1 - given2} + {bought} = {rem}.\n"
            f"#### {rem}"
        )
        items.append({
            "question": q,
            "answer": a,
            "category": "cot_arithmetic",
            "is_direct_slot": False,
            "has_anchor": True,
        })

    # 2. Temporal & Clock State-Machine Reasoning (1,500)
    for _ in range(1500):
        choice = random.random()
        if choice < 0.5:
            # Clock addition
            h_start = random.randint(1, 12)
            m_start = random.choice([0, 15, 30, 45, 50])
            ampm = random.choice(["AM", "PM"])
            h_dur = random.randint(1, 7)
            m_dur = random.choice([15, 30, 45])

            tot_min = (h_start % 12) * 60 + m_start + (720 if ampm == "PM" else 0) + (h_dur * 60 + m_dur)
            tot_min = tot_min % 1440
            h_end = tot_min // 60
            m_end = tot_min % 60
            ampm_end = "PM" if h_end >= 12 else "AM"
            h_disp = 12 if (h_end % 12 == 0) else (h_end % 12)
            res_str = f"{h_disp}:{m_end:02d} {ampm_end}"

            q = f"A journey begins at {h_start}:{m_start:02d} {ampm} and lasts for {h_dur} hour(s) and {m_dur} minute(s). What time does it conclude?"
            a = (
                f"1. Start time is {h_start}:{m_start:02d} {ampm}.\n"
                f"2. Add {h_dur} hours to the departure time.\n"
                f"3. Add the remaining {m_dur} minutes, carrying over to the next hour if needed.\n"
                f"4. The arrival time is {res_str}.\n"
                f"#### {res_str}"
            )
        else:
            # Day offset
            day_idx = random.randint(0, 6)
            offset = random.randint(2, 6)
            target_idx = (day_idx + offset) % 7
            day_now = DAYS[day_idx]
            day_ans = DAYS[target_idx]

            q = f"If today is {day_now}, what day of the week will it be in exactly {offset} days?"
            a = (
                f"1. Today is {day_now} (Day {day_idx + 1} of the cycle).\n"
                f"2. Count forward {offset} day(s): ({day_idx} + {offset}) mod 7 = {target_idx}.\n"
                f"3. Day index {target_idx} corresponds to {day_ans}.\n"
                f"#### {day_ans}"
            )

        items.append({
            "question": q,
            "answer": a,
            "category": "cot_temporal",
            "is_direct_slot": False,
            "has_anchor": True,
        })

    # 3. Counterfactual Premise Rejection (1,000)
    cf_data = [
        ("What is the capital city of Antarctica?", "Antarctica does not have a capital city. It is a continent governed by international scientific treaties with no sovereign government or permanent native capital. #### None"),
        ("How many legs does a three-legged dog have?", "1. The premise explicitly specifies that the dog is three-legged.\n2. Therefore, the dog has exactly 3 legs.\n#### 3"),
        ("If all squares are circles and all circles are triangles, are all squares triangles?", "1. Given: All squares are circles (S ⊆ C).\n2. Given: All circles are triangles (C ⊆ T).\n3. By transitive deduction, S ⊆ T.\n4. Therefore, all squares are triangles.\n#### Yes"),
        ("Can an airplane fly backward in calm air under its own jet thrust?", "Commercial and standard jet airplanes generate lift via forward airspeed over their wings. In calm air, they cannot sustain flight backward under their own thrust. #### No"),
        ("What color is the black box on a commercial airliner?", "1. Despite being called a 'black box', flight data recorders are painted bright fluorescent orange for visibility in crash recovery.\n2. Therefore, the color is orange.\n#### Orange"),
    ]
    for _ in range(1000):
        q, a = random.choice(cf_data)
        items.append({
            "question": q,
            "answer": a,
            "category": "cot_counterfactual",
            "is_direct_slot": False,
            "has_anchor": True,
        })

    random.shuffle(items)
    return items[:target_count]


# ==============================================================================
# PILLAR 1C: INVARIANT SCIENCE & AGENTIC ANCHORS (1,500 EXAMPLES - 15%)
# ==============================================================================

SCIENCE_TOPICS = [
    ("Why is the sky blue during the day?", "The sky appears blue because molecules in Earth's atmosphere scatter shorter wavelengths of sunlight (blue and violet) much more efficiently than longer wavelengths (red and yellow), a phenomenon known as Rayleigh scattering."),
    ("Why does the sky turn red at sunset?", "At sunset, sunlight must travel through a much greater thickness of atmosphere. The short blue wavelengths are scattered away before reaching the observer, leaving the longer red and orange wavelengths to pass through."),
    ("What are Newton's three laws of motion?", "1. First Law (Inertia): An object remains at rest or in uniform motion unless acted upon by an external force.\n2. Second Law (F = ma): Acceleration is proportional to net force and inversely proportional to mass.\n3. Third Law: For every action, there is an equal and opposite reaction."),
    ("What is the principle of conservation of energy?", "The principle of conservation of energy states that energy can neither be created nor destroyed; it can only be transformed from one form into another, so the total energy of an isolated system remains constant."),
    ("What is the difference between speed and velocity?", "Speed is a scalar quantity measuring the rate at which an object covers distance (magnitude only), while velocity is a vector quantity specifying both speed and direction of motion."),
    ("How does an enzyme catalyze a chemical reaction?", "An enzyme catalyzes a chemical reaction by binding to substrate molecules at its active site, stabilizing the transition state and significantly lowering the activation energy required for the reaction to proceed."),
    ("What is the central dogma of molecular biology?", "The central dogma describes the flow of genetic information: DNA is transcribed into messenger RNA (mRNA), which is then translated by ribosomes into functional proteins (DNA -> RNA -> Protein)."),
]

TOOL_TEMPLATES = [
    ("Search for scientific papers on CRISPR gene editing published in 2025:", "action: academic_search(query='CRISPR gene editing', year=2025, limit=5)"),
    ("Calculate the square root of 144 multiplied by 15:", "action: calculate(expression='sqrt(144) * 15')"),
    ("Look up the weather forecast for Tokyo tomorrow:", "action: get_weather(location='Tokyo', timeframe='tomorrow')"),
    ("Check database records for customer ID 94821:", "action: query_database(table='customers', filter={'id': 94821})"),
    ("Schedule a team sync meeting on Friday at 2:00 PM for 30 minutes:", "action: calendar_schedule(title='Team Sync', day='Friday', time='14:00', duration_minutes=30)"),
    ("Fetch the latest currency exchange rate between EUR and USD:", "action: get_forex_rate(base='EUR', target='USD')"),
    ("Execute Python code to generate the first 10 Fibonacci numbers:", "action: run_code(language='python', script='fib = [0, 1]; [fib.append(fib[-1] + fib[-2]) for _ in range(8)]; print(fib)')"),
]


def generate_science_and_agentic_anchors(target_count: int = 1500) -> List[Dict[str, Any]]:
    """Generates 1,500 Invariant Science & Agentic structured tool dispatch traces."""
    items = []

    # 1. Science Q&A (750)
    for _ in range(750):
        q, a = random.choice(SCIENCE_TOPICS)
        items.append({
            "question": q,
            "answer": a,
            "category": "science_anchor",
            "is_direct_slot": False,
            "has_anchor": False,
        })

    # 2. Agentic Tool Dispatch Traces (750)
    for _ in range(750):
        q, a = random.choice(TOOL_TEMPLATES)
        items.append({
            "question": q,
            "answer": a,
            "category": "agentic_tool",
            "is_direct_slot": True,
            "has_anchor": False,
        })

    random.shuffle(items)
    return items[:target_count]


# ==============================================================================
# MAIN COMPILED DATASET BUILDER
# ==============================================================================

def build_tri_modal_dataset():
    print("=" * 80)
    print("🧠 CEREBROSPHEROID V5: REFINED TRI-MODAL SFT DATASET BUILDER")
    print("=" * 80)

    print("\n[Track 1/3] Synthesizing 4,500 Direct Instruction & Slot QA pairs (45%)...")
    direct_qa = generate_direct_slot_qa(4500)
    print(f"   ✓ Generated {len(direct_qa):,} Direct Slot QA pairs.")

    print("\n[Track 2/3] Synthesizing 4,000 Full-Horizon Cognitive CoT problems (40%)...")
    cot_qa = generate_cognitive_cot(4000)
    print(f"   ✓ Generated {len(cot_qa):,} Full-Horizon CoT problems.")

    print("\n[Track 3/3] Synthesizing 1,500 Science & Agentic Tool Anchors (15%)...")
    anchors_qa = generate_science_and_agentic_anchors(1500)
    print(f"   ✓ Generated {len(anchors_qa):,} Science & Agentic Anchors.")

    all_data = direct_qa + cot_qa + anchors_qa
    random.shuffle(all_data)
    total_records = len(all_data)
    print(f"\n📊 Total Tri-Modal SFT Corpus: {total_records:,} High-Density Records")

    # Output directory
    out_dir = os.path.join(ROOT_DIR, "data", "cerebro_cognitive_8k")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Write Parquet
    parquet_path = os.path.join(out_dir, "cerebro_tri_modal_sft_10k.parquet")
    legacy_parquet = os.path.join(out_dir, "cerebro_cognitive_10k.parquet")

    table = pa.Table.from_pydict({
        "question": [d["question"] for d in all_data],
        "answer": [d["answer"] for d in all_data],
        "category": [d["category"] for d in all_data],
        "is_direct_slot": [d["is_direct_slot"] for d in all_data],
        "has_anchor": [d["has_anchor"] for d in all_data],
    })
    pq.write_table(table, parquet_path)
    pq.write_table(table, legacy_parquet)
    print(f"   ✓ Saved Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")
    print(f"   ✓ Updated Base:  {legacy_parquet} ({os.path.getsize(legacy_parquet) / 1024:.1f} KB)")

    # 2. Write JSONL (resolves missing jsonl issue)
    jsonl_path = os.path.join(out_dir, "cerebro_tri_modal_sft_10k.jsonl")
    legacy_jsonl = os.path.join(out_dir, "cerebro_cognitive_10k.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for d in all_data:
            f_out.write(json.dumps(d) + "\n")

    with open(legacy_jsonl, "w", encoding="utf-8") as f_out:
        for d in all_data:
            f_out.write(json.dumps(d) + "\n")

    print(f"   ✓ Saved JSONL:   {jsonl_path} ({os.path.getsize(jsonl_path) / 1024:.1f} KB)")
    print(f"   ✓ Updated Base:  {legacy_jsonl} ({os.path.getsize(legacy_jsonl) / 1024:.1f} KB)")

    print("\n" + "=" * 80)
    print("✨ TRI-MODAL SFT DATASET GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    build_tri_modal_dataset()
