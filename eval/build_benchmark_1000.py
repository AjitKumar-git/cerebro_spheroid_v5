"""
Cerebro-1000 Benchmark Dataset Builder
======================================
Compiles 1,000 standardized, high-quality evaluation prompts designed to rigorously
benchmark CerebroSpheroid models across 5 fundamental cognitive pillars (200 prompts each):

1. linguistic_and_syntax: Grammar, completion, analogies, vocabulary, orthography.
2. factual_and_counterfactual: Science, history, geography, false premise rejection.
3. mathematical_and_logic: Multi-step arithmetic, algebra, temporal logic, deduction.
4. code_and_algorithms: Python logic, recursion, debugging, complexity, regex.
5. needle_and_agentic: Needle-in-haystack sprinkled retrieval, tool selection, multi-constraint tasks.
"""

import json
import os
import random
import re

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_1000.jsonl")


def build_linguistic_pillar():
    """Generates 200 linguistic, syntax, and language modeling prompts."""
    items = []
    
    # 1. Analogies & Antonyms (40)
    analogy_pairs = [
        ("hot", "cold", "day", "night"),
        ("bird", "fly", "fish", "swim"),
        ("author", "book", "artist", "painting"),
        ("doctor", "hospital", "teacher", "school"),
        ("glove", "hand", "sock", "foot"),
        ("pen", "write", "knife", "cut"),
        ("lion", "carnivore", "deer", "herbivore"),
        ("sun", "day", "moon", "night"),
        ("tree", "forest", "sand", "desert"),
        ("water", "thirst", "food", "hunger"),
        ("wheel", "car", "wing", "airplane"),
        ("key", "lock", "password", "account"),
        ("rain", "flood", "drought", "famine"),
        ("caterpillar", "butterfly", "tadpole", "frog"),
        ("conductor", "orchestra", "director", "movie"),
        ("hammer", "nail", "screwdriver", "screw"),
        ("telescope", "star", "microscope", "cell"),
        ("ship", "captain", "plane", "pilot"),
        ("seed", "plant", "egg", "bird"),
        ("needle", "thread", "bow", "arrow"),
    ]
    for i, (a1, a2, b1, b2) in enumerate(analogy_pairs):
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "analogy",
            "difficulty": "easy",
            "prompt": f"Complete the analogy: {a1} is to {a2} as {b1} is to ____.",
            "reference_answer": b2,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(b2)}\b"
        })
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "antonym",
            "difficulty": "easy",
            "prompt": f"What is the exact antonym of the word '{a1}'?",
            "reference_answer": a2,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(a2)}\b"
        })

    # 2. Grammar & Tense Transformation (40)
    verbs = [
        ("run", "ran", "running"),
        ("write", "wrote", "writing"),
        ("sing", "sang", "singing"),
        ("swim", "swam", "swimming"),
        ("fly", "flew", "flying"),
        ("eat", "ate", "eating"),
        ("break", "broke", "breaking"),
        ("drive", "drove", "driving"),
        ("speak", "spoke", "speaking"),
        ("begin", "began", "beginning"),
        ("drink", "drank", "drinking"),
        ("fall", "fell", "falling"),
        ("give", "gave", "giving"),
        ("know", "knew", "knowing"),
        ("see", "saw", "seeing"),
        ("take", "took", "taking"),
        ("wear", "wore", "wearing"),
        ("grow", "grew", "growing"),
        ("hide", "hid", "hiding"),
        ("choose", "chose", "choosing"),
    ]
    for base, past, part in verbs:
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "grammar_tense",
            "difficulty": "easy",
            "prompt": f"What is the simple past tense of the irregular verb '{base}'?",
            "reference_answer": past,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(past)}\b"
        })
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "grammar_participle",
            "difficulty": "easy",
            "prompt": f"What is the present participle (-ing form) of '{base}'?",
            "reference_answer": part,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(part)}\b"
        })

    # 3. Orthography & Spelling Verification (40)
    spelling_words = [
        ("accommodate", "accommodate", ["accomodate", "acommodate"]),
        ("necessary", "necessary", ["neccessary", "necesary"]),
        ("separate", "separate", ["seperate"]),
        ("definitely", "definitely", ["definatly", "definitly"]),
        ("rhythm", "rhythm", ["rythm", "rhythym"]),
        ("embarrass", "embarrass", ["embarass"]),
        ("occurrence", "occurrence", ["occurence", "occurrance"]),
        ("millennium", "millennium", ["millenium"]),
        ("privilege", "privilege", ["privelege", "priviledge"]),
        ("liaison", "liaison", ["liason"]),
        ("conscientious", "conscientious", ["consciencious"]),
        ("hierarchy", "hierarchy", ["heirarchy"]),
        ("maintenance", "maintenance", ["maintainance"]),
        ("bureaucracy", "bureaucracy", ["bureacracy"]),
        ("consensus", "consensus", ["concensus"]),
        ("perseverance", "perseverance", ["perseverence"]),
        ("weird", "weird", ["wierd"]),
        ("supersede", "supersede", ["supercede"]),
        ("pronunciation", "pronunciation", ["pronounciation"]),
        ("gauge", "gauge", ["guage"]),
    ]
    for correct, answer, wrong_list in spelling_words:
        w = wrong_list[0]
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "orthography",
            "difficulty": "medium",
            "prompt": f"Between '{correct}' and '{w}', which one is the correct standard English spelling?",
            "reference_answer": correct,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(correct)}\b"
        })
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "spelling_reversal",
            "difficulty": "medium",
            "prompt": f"Spell the word '{correct[:6]}' in reverse order, letter by letter.",
            "reference_answer": correct[:6][::-1],
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(correct[:6][::-1])}\b"
        })

    # 4. Lexical Constraints & Sentence Completion (40)
    constraints = [
        ("Write a 5-word sentence where every word starts with the letter 's'.", "six silent snakes slither slowly", r"(?i)\bs[a-z]+\s+s[a-z]+\s+s[a-z]+\s+s[a-z]+\s+s[a-z]+\b"),
        ("Complete the famous idiom: 'A bird in the hand is worth two in the ____.'", "bush", r"(?i)\bbush\b"),
        ("Complete the saying: 'Actions speak louder than ____.'", "words", r"(?i)\bwords\b"),
        ("Complete the idiom: 'Bite the ____.'", "bullet", r"(?i)\bbullet\b"),
        ("Complete the proverb: 'Every cloud has a silver ____.'", "lining", r"(?i)\blining\b"),
        ("Complete the saying: 'When in Rome, do as the ____ do.'", "Romans", r"(?i)\bRomans\b"),
        ("Complete the expression: 'Piece of ____.'", "cake", r"(?i)\bcake\b"),
        ("Complete the idiom: 'The ball is in your ____.'", "court", r"(?i)\bcourt\b"),
        ("Complete the proverb: 'Practice makes ____.'", "perfect", r"(?i)\bperfect\b"),
        ("Complete the phrase: 'Curiosity killed the ____.'", "cat", r"(?i)\bcat\b"),
        ("What punctuation mark is used to connect two independent clauses without a conjunction?", "semicolon", r"(?i)\bsemicolon\b"),
        ("What punctuation mark indicates omission of letters or possession?", "apostrophe", r"(?i)\bapostrophe\b"),
        ("What is a word called that reads the same backward as forward, such as 'radar'?", "palindrome", r"(?i)\bpalindrome\b"),
        ("What is a word formed from the initial letters of other words, such as NASA?", "acronym", r"(?i)\bacronym\b"),
        ("What figure of speech makes an explicit comparison using 'like' or 'as'?", "simile", r"(?i)\bsimile\b"),
        ("What figure of speech attributes human characteristics to non-human things?", "personification", r"(?i)\bpersonification\b"),
        ("What is an exaggerated statement not meant to be taken literally called?", "hyperbole", r"(?i)\bhyperbole\b"),
        ("What is a word that sounds like what it describes, such as 'buzz' or 'sizzle'?", "onomatopoeia", r"(?i)\bonomatopoeia\b"),
        ("What is the term for two contradictory words placed together, such as 'deafening silence'?", "oxymoron", r"(?i)\boxymoron\b"),
        ("What is a sentence containing all 26 letters of the English alphabet called?", "pangram", r"(?i)\bpangram\b"),
    ]
    for prompt, ans, reg in constraints:
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "lexical_semantics",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": reg
        })
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "vocabulary_precision",
            "difficulty": "medium",
            "prompt": f"Explain briefly what is meant by '{ans}' in English literature or grammar.",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(ans[:6])}\b"
        })

    # 5. Semantic Synonyms & Nuance (40)
    synonym_tuples = [
        ("benevolent", "kind", "malevolent"),
        ("meticulous", "careful", "careless"),
        ("candid", "honest", "deceitful"),
        ("ephemeral", "temporary", "permanent"),
        ("lucid", "clear", "confusing"),
        ("gregarious", "sociable", "solitary"),
        ("pragmatic", "practical", "idealistic"),
        ("resilient", "tough", "fragile"),
        ("tenacious", "persistent", "yielding"),
        ("voracious", "insatiable", "apathetic"),
        ("ubiquitous", "omnipresent", "rare"),
        ("austere", "strict", "luxurious"),
        ("cacophony", "harsh noise", "harmony"),
        ("diligent", "hardworking", "lazy"),
        ("fervent", "passionate", "indifferent"),
        ("immutable", "unchangeable", "variable"),
        ("nefarious", "wicked", "virtuous"),
        ("prolific", "productive", "unfruitful"),
        ("stoic", "unemotional", "expressive"),
        ("zealous", "enthusiastic", "apathetic"),
    ]
    for word, syn, ant in synonym_tuples:
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "synonym",
            "difficulty": "hard",
            "prompt": f"Provide the closest common single-word synonym for the advanced adjective '{word}'.",
            "reference_answer": syn,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(syn.split()[0])}\b"
        })
        items.append({
            "category": "linguistic_and_syntax",
            "sub_category": "antonym_advanced",
            "difficulty": "hard",
            "prompt": f"What is the antonym of '{word}'?",
            "reference_answer": ant,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{re.escape(ant)}\b"
        })

    return items[:200]


def build_factual_pillar():
    """Generates 200 factual, scientific, and counterfactual disambiguation prompts."""
    items = []

    # 1. Counterfactual Premise Rejection (40)
    counterfactuals = [
        ("What is the capital city of Antarctica?", "Antarctica has no capital city or government.", r"(?i)(no capital|does not have|none)"),
        ("Who was the President of the United States in the year 1650?", "The United States did not exist in 1650.", r"(?i)(did not exist|no president|not founded)"),
        ("How many rings does the planet Mercury have?", "Mercury has zero rings.", r"(?i)(zero|no rings|none|\b0\b)"),
        ("What year did Neil Armstrong walk on the surface of Mars?", "Neil Armstrong never walked on Mars (he walked on the Moon in 1969).", r"(?i)(never|moon|did not walk on mars)"),
        ("What is the boiling point of pure water in degrees Kelvin at 1 atmosphere?", "373.15 K", r"(?i)(373|373\.15)"),
        ("Which ocean separates Africa from South America?", "Atlantic Ocean", r"(?i)Atlantic"),
        ("What is the chemical formula for ordinary table salt?", "NaCl", r"(?i)NaCl"),
        ("How many chromosomes are in a typical human diploid somatic cell?", "46", r"\b46\b"),
        ("What is the powerhouse organelle of eukaryotic cells responsible for ATP production?", "mitochondria", r"(?i)mitochon"),
        ("Which element has atomic number 1 on the periodic table?", "Hydrogen", r"(?i)Hydrogen"),
        ("What force keeps planets in orbit around the Sun?", "Gravity", r"(?i)gravit"),
        ("What is the speed of light in vacuum approximately in kilometers per second?", "300,000 km/s", r"(?i)(300[, ]?000|299[, ]?792)"),
        ("What gas do green plants absorb from the atmosphere during photosynthesis?", "Carbon dioxide", r"(?i)(carbon dioxide|CO2)"),
        ("What is the hardest naturally occurring mineral on Earth?", "Diamond", r"(?i)diamond"),
        ("Which planet in our solar system is known as the Red Planet?", "Mars", r"(?i)Mars"),
        ("What is the primary function of red blood cells in the human body?", "Transport oxygen", r"(?i)(oxygen|O2)"),
        ("What is the capital city of Australia?", "Canberra", r"(?i)Canberra"),
        ("What is the longest river in the world by general consensus?", "Nile", r"(?i)Nile"),
        ("In what year did the Titanic sink?", "1912", r"\b1912\b"),
        ("Who developed the theory of general relativity?", "Albert Einstein", r"(?i)Einstein"),
    ]
    for prompt, ans, reg in counterfactuals:
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "counterfactual_rejection",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "scientific_verification",
            "difficulty": "medium",
            "prompt": f"Verify the truth value of the statement: '{prompt}' and state the definitive answer clearly.",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 2. Physics & Chemistry Principles (40)
    science_facts = [
        ("What is Newton's first law of motion commonly referred to as?", "Law of inertia", r"(?i)inertia"),
        ("What subatomic particle carries a negative electric charge?", "Electron", r"(?i)electron"),
        ("What type of bond is formed when atoms share electron pairs?", "Covalent bond", r"(?i)covalent"),
        ("What is the SI unit of electric resistance?", "Ohm", r"(?i)ohm"),
        ("What is the SI unit of frequency?", "Hertz", r"(?i)hertz"),
        ("What is the pH of pure neutral water at 25 degrees Celsius?", "7", r"\b7(?:\.0)?\b"),
        ("Which noble gas is the lightest?", "Helium", r"(?i)helium"),
        ("What state of matter consists of ionized gas with free electrons?", "Plasma", r"(?i)plasma"),
        ("What is the acceleration due to gravity on Earth's surface approximately in m/s^2?", "9.8 m/s^2", r"(?i)9\.8"),
        ("What law states that energy cannot be created or destroyed, only transformed?", "First law of thermodynamics / Conservation of energy", r"(?i)(conservation of energy|first law of thermodynamics)"),
        ("What particle is composed of two up quarks and one down quark?", "Proton", r"(?i)proton"),
        ("What is the half-life of Carbon-14 approximately in years?", "5,730 years", r"(?i)5,?730"),
        ("What electromagnetic radiation has wavelengths just shorter than visible violet light?", "Ultraviolet", r"(?i)ultraviolet"),
        ("What is the absolute zero temperature in degrees Celsius?", "-273.15", r"(?i)-273(\.15)?"),
        ("What device converts mechanical energy into electrical energy?", "Generator", r"(?i)generator"),
        ("What is the process called when a solid turns directly into a gas without melting?", "Sublimation", r"(?i)sublimation"),
        ("Which element is the most abundant in the Earth's crust by mass?", "Oxygen", r"(?i)oxygen"),
        ("What is the atomic number of Carbon?", "6", r"\b6\b"),
        ("What is the primary gas found in the Earth's atmosphere?", "Nitrogen", r"(?i)nitrogen"),
        ("What law relates pressure and volume of an ideal gas at constant temperature?", "Boyle's Law", r"(?i)Boyle"),
    ]
    for prompt, ans, reg in science_facts:
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "physics_chemistry",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "physics_chemistry_definition",
            "difficulty": "medium",
            "prompt": f"State precisely: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 3. Biology & Genetics (40)
    bio_facts = [
        ("What molecule carries genetic instructions in all living organisms?", "DNA", r"(?i)\bDNA\b"),
        ("What base pairs with Adenine in DNA?", "Thymine", r"(?i)thymine"),
        ("What base replaces Thymine in RNA?", "Uracil", r"(?i)uracil"),
        ("What blood type is known as the universal red cell donor?", "O negative", r"(?i)O\s*(?:negative|-|neg)"),
        ("How many chambers are in the human heart?", "4", r"\b4\b"),
        ("Which organ in the human body produces insulin?", "Pancreas", r"(?i)pancreas"),
        ("What is the process of cell division that results in four genetically diverse daughter cells?", "Meiosis", r"(?i)meiosis"),
        ("What is the largest organ of the human body by surface area?", "Skin", r"(?i)skin"),
        ("What kingdom do mushrooms and yeasts belong to?", "Fungi", r"(?i)fungi"),
        ("What is the protein that carries oxygen in human red blood cells?", "Hemoglobin", r"(?i)hemoglobin"),
        ("What part of the brain controls balance and coordination?", "Cerebellum", r"(?i)cerebellum"),
        ("What type of macromolecule are enzymes predominantly made of?", "Proteins", r"(?i)protein"),
        ("What is the basic unit of heredity?", "Gene", r"(?i)gene"),
        ("What vascular tissue in plants transports water from roots to leaves?", "Xylem", r"(?i)xylem"),
        ("What vascular tissue in plants transports sugars and nutrients from leaves?", "Phloem", r"(?i)phloem"),
        ("What is the technical term for cold-blooded animals whose body temperature depends on external heat?", "Ectothermic", r"(?i)ectotherm"),
        ("What structure protects bacterial cells and gives them shape?", "Cell wall", r"(?i)cell wall"),
        ("What is the standard start codon in messenger RNA translation?", "AUG", r"(?i)AUG"),
        ("Which amino acid is encoded by the AUG start codon?", "Methionine", r"(?i)methionine"),
        ("What organ filters waste products from blood to produce urine?", "Kidney", r"(?i)kidney"),
    ]
    for prompt, ans, reg in bio_facts:
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "biology_genetics",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "biology_genetics_quiz",
            "difficulty": "medium",
            "prompt": f"Identify the biological concept: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 4. Geography & Earth Science (40)
    geo_facts = [
        ("What is the deepest oceanic trench on Earth?", "Mariana Trench", r"(?i)Mariana"),
        ("What is the highest mountain peak in North America?", "Denali (Mount McKinley)", r"(?i)Denali"),
        ("Which continent contains the Sahara Desert?", "Africa", r"(?i)Africa"),
        ("What is the capital city of Canada?", "Ottawa", r"(?i)Ottawa"),
        ("What is the capital city of Japan?", "Tokyo", r"(?i)Tokyo"),
        ("What is the smallest country in the world by both area and population?", "Vatican City", r"(?i)Vatican"),
        ("What straight connects the Atlantic Ocean to the Mediterranean Sea?", "Strait of Gibraltar", r"(?i)Gibraltar"),
        ("Which tectonic boundary is responsible for the San Andreas Fault?", "Transform boundary", r"(?i)transform"),
        ("What is the largest freshwater lake in the world by surface area?", "Lake Superior", r"(?i)Superior"),
        ("What layer of the Earth lies directly beneath the crust?", "Mantle", r"(?i)mantle"),
        ("Which river flows through Paris, France?", "Seine", r"(?i)Seine"),
        ("What is the capital of Brazil?", "Brasilia", r"(?i)Brasilia"),
        ("Which country has the longest coastline in the world?", "Canada", r"(?i)Canada"),
        ("What desert is located in northern Chile and is one of the driest places on Earth?", "Atacama", r"(?i)Atacama"),
        ("Which country possesses the island of Greenland as an autonomous territory?", "Denmark", r"(?i)Denmark"),
        ("What is the capital city of Egypt?", "Cairo", r"(?i)Cairo"),
        ("What mountain range separates Europe from Asia?", "Ural Mountains", r"(?i)Ural"),
        ("What is the currency of the United Kingdom?", "Pound Sterling", r"(?i)pound"),
        ("What is the capital city of South Korea?", "Seoul", r"(?i)Seoul"),
        ("What is the southernmost continent on Earth?", "Antarctica", r"(?i)Antarctica"),
    ]
    for prompt, ans, reg in geo_facts:
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "geography",
            "difficulty": "easy",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "geography_capital",
            "difficulty": "easy",
            "prompt": f"Answer accurately: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 5. History & Chronology (40)
    history_facts = [
        ("In what year did World War II end?", "1945", r"\b1945\b"),
        ("In what year did the United States declare independence from Great Britain?", "1776", r"\b1776\b"),
        ("Who was the first emperor of the Roman Empire?", "Augustus", r"(?i)Augustus"),
        ("In what year did the Berlin Wall fall?", "1989", r"\b1989\b"),
        ("Which civilization built the pyramids of Giza?", "Ancient Egyptians", r"(?i)Egypt"),
        ("In what year did Christopher Columbus first reach the Americas?", "1492", r"\b1492\b"),
        ("Who was the British Prime Minister during most of World War II?", "Winston Churchill", r"(?i)Churchill"),
        ("What war was fought between the American North and South from 1861 to 1865?", "American Civil War", r"(?i)Civil War"),
        ("In what year was the Magna Carta signed by King John of England?", "1215", r"\b1215\b"),
        ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo", r"(?i)Michelangelo"),
        ("In what year did the French Revolution begin with the storming of the Bastille?", "1789", r"\b1789\b"),
        ("Who wrote the Communist Manifesto alongside Friedrich Engels in 1848?", "Karl Marx", r"(?i)Marx"),
        ("What empire fell when Constantinople was captured in 1453?", "Byzantine Empire", r"(?i)Byzantine"),
        ("Who was the first female Prime Minister of the United Kingdom?", "Margaret Thatcher", r"(?i)Thatcher"),
        ("In what year was the United Nations established?", "1945", r"\b1945\b"),
        ("Who led the Salt March in India in 1930 as an act of nonviolent civil disobedience?", "Mahatma Gandhi", r"(?i)Gandhi"),
        ("What ancient city was destroyed and preserved by the eruption of Mount Vesuvius in 79 AD?", "Pompeii", r"(?i)Pompeii"),
        ("What treaty formally ended World War I in 1919?", "Treaty of Versailles", r"(?i)Versailles"),
        ("Who was the principal author of the United States Declaration of Independence?", "Thomas Jefferson", r"(?i)Jefferson"),
        ("In what year did the Soviet Union officially dissolve?", "1991", r"\b1991\b"),
    ]
    for prompt, ans, reg in history_facts:
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "history_chronology",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "factual_and_counterfactual",
            "sub_category": "history_quiz",
            "difficulty": "medium",
            "prompt": f"State the historical fact: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    return items[:200]


def build_math_logic_pillar():
    """Generates 200 mathematical, arithmetic, and logical reasoning prompts."""
    items = []

    # 1. Direct Multi-Step Arithmetic (50)
    for i in range(1, 26):
        a = 12 + i * 3
        b = 7 + i * 2
        c = 4 + (i % 5)
        ans1 = a * b - c
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "arithmetic_multistep",
            "difficulty": "easy",
            "prompt": f"Calculate the exact numerical value of: ({a} * {b}) - {c}.",
            "reference_answer": str(ans1),
            "expected_action": "calculator",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans1}\b"
        })
        d = 200 + i * 15
        e = 5 + (i % 4)
        ans2 = (d // e) + a
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "arithmetic_division",
            "difficulty": "easy",
            "prompt": f"Compute the integer expression: ({d} // {e}) + {a}.",
            "reference_answer": str(ans2),
            "expected_action": "calculator",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans2}\b"
        })

    # 2. GSM8K Word Problems (50)
    gsm8k_templates = [
        ("Janet's ducks lay {d} eggs per day. She eats {e} for breakfast and bakes muffins with {m}. She sells the rest for ${p} per egg. How much money in dollars does she make each day?",
         lambda d, e, m, p: (d - e - m) * p),
        ("A bakery sells cupcakes for ${p} each. Today, they baked {b} cupcakes. By noon, they sold {s} cupcakes. How much revenue did they make from the cupcakes sold by noon?",
         lambda p, b, s: s * p),
        ("Tom has {c} cars. Each car has 4 wheels and 1 spare tire in the trunk. How many total tires do Tom's cars have altogether?",
         lambda c: c * 5),
        ("Sarah read {p1} pages of her book on Monday, {p2} pages on Tuesday, and {p3} pages on Wednesday. If the book has {tot} pages, how many pages does she have left to read?",
         lambda p1, p2, p3, tot: tot - (p1 + p2 + p3)),
        ("An apple orchard has {r} rows of trees. Each row contains {t} apple trees. If each tree produces {a} apples, how many total apples are produced?",
         lambda r, t, a: r * t * a),
    ]
    for i in range(10):
        # 5 problems per iteration = 50 total
        # Prob 1
        d, e, m, p = 16 + i, 3, 4, 2
        ans = (d - e - m) * p
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "gsm8k_word_problem",
            "difficulty": "medium",
            "prompt": f"Janet's ducks lay {d} eggs per day. She eats {e} for breakfast and bakes muffins with {m}. She sells the rest for ${p} per egg. How much money in dollars does she make each day?",
            "reference_answer": str(ans),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans}\b"
        })
        # Prob 2
        price, baked, sold = 4, 50 + i * 5, 20 + i * 2
        ans = sold * price
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "gsm8k_word_problem",
            "difficulty": "medium",
            "prompt": f"A bakery sells cupcakes for ${price} each. Today, they baked {baked} cupcakes. By noon, they sold {sold} cupcakes. How much revenue in dollars did they make from the cupcakes sold by noon?",
            "reference_answer": str(ans),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans}\b"
        })
        # Prob 3
        cars = 6 + i
        ans = cars * 5
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "gsm8k_word_problem",
            "difficulty": "medium",
            "prompt": f"Tom has {cars} cars. Each car has 4 road tires and 1 spare tire in the trunk. How many total tires do Tom's cars have altogether?",
            "reference_answer": str(ans),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans}\b"
        })
        # Prob 4
        p1, p2, p3, tot = 25 + i, 30 + i, 45, 150 + i * 5
        ans = tot - (p1 + p2 + p3)
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "gsm8k_word_problem",
            "difficulty": "medium",
            "prompt": f"Sarah read {p1} pages of her book on Monday, {p2} pages on Tuesday, and {p3} pages on Wednesday. If the book has {tot} pages, how many pages does she have left to read?",
            "reference_answer": str(ans),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans}\b"
        })
        # Prob 5
        r, t, a = 5 + i, 8, 12
        ans = r * t * a
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "gsm8k_word_problem",
            "difficulty": "medium",
            "prompt": f"An apple orchard has {r} rows of trees. Each row contains {t} apple trees. If each tree produces {a} apples, how many total apples are produced in the orchard?",
            "reference_answer": str(ans),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{ans}\b"
        })

    # 3. Temporal State Machine (50)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(25):
        d_idx = i % 7
        curr_day = days[d_idx]
        # yesterday was curr_day - 1, tomorrow is curr_day + 1
        y_day = days[(d_idx - 1) % 7]
        t_day = days[(d_idx + 1) % 7]
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "temporal_logic",
            "difficulty": "easy",
            "prompt": f"If yesterday was {y_day}, what day of the week is tomorrow?",
            "reference_answer": t_day,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{t_day}\b"
        })
        day_after_tomorrow = days[(d_idx + 2) % 7]
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "temporal_logic",
            "difficulty": "medium",
            "prompt": f"Today is {curr_day}. What day will it be 10 days from today?",
            "reference_answer": days[(d_idx + 10) % 7],
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{days[(d_idx + 10) % 7]}\b"
        })

    # 4. Elementary Algebra & Equations (50)
    for i in range(1, 26):
        # 2x + b = c -> x = (c - b) / 2
        x_val = 3 + i
        b_val = 5 + (i % 7)
        c_val = 2 * x_val + b_val
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "algebra",
            "difficulty": "medium",
            "prompt": f"Solve for x in the linear algebraic equation: 2x + {b_val} = {c_val}.",
            "reference_answer": str(x_val),
            "expected_action": "calculator",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{x_val}\b"
        })
        # 3y - d = e -> y = (e + d) / 3
        y_val = 4 + i
        d_val = 7 + (i % 5)
        e_val = 3 * y_val - d_val
        items.append({
            "category": "mathematical_and_logic",
            "sub_category": "algebra",
            "difficulty": "medium",
            "prompt": f"Solve for y: 3y - {d_val} = {e_val}.",
            "reference_answer": str(y_val),
            "expected_action": "calculator",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{y_val}\b"
        })

    return items[:200]


def build_code_pillar():
    """Generates 200 code, Python logic, algorithmic, and debugging prompts."""
    items = []

    # 1. Python Syntax & Execution (50)
    code_snippets = [
        ("What is the output of `len([x for x in range(10) if x % 2 == 0])` in Python?", "5", r"\b5\b"),
        ("What does `sorted([5, 2, 9, 1])[1]` evaluate to in Python?", "2", r"\b2\b"),
        ("What is the return value of `'hello world'.split()[0]` in Python?", "'hello'", r"(?i)['\"]?hello['\"]?"),
        ("What does `list(range(2, 8, 2))` produce in Python?", "[2, 4, 6]", r"\[2,\s*4,\s*6\]"),
        ("What does `bool('')` evaluate to in Python?", "False", r"(?i)\bFalse\b"),
        ("What does `bool([0])` evaluate to in Python?", "True", r"(?i)\bTrue\b"),
        ("What built-in function returns both the index and value when iterating over an iterable?", "enumerate", r"(?i)\benumerate\b"),
        ("What built-in function pairs corresponding elements from two or more iterables?", "zip", r"(?i)\bzip\b"),
        ("What method removes and returns the last element from a Python list?", "pop", r"(?i)\bpop\b"),
        ("What dictionary method retrieves a value for a key without raising a KeyError if absent?", "get", r"(?i)\bget\b"),
        ("What keyword is used in Python to define an anonymous inline function?", "lambda", r"(?i)\blambda\b"),
        ("What keyword defines a generator function that yields items one at a time?", "yield", r"(?i)\byield\b"),
        ("What keyword is used to ensure code executes after a try-except block regardless of errors?", "finally", r"(?i)\bfinally\b"),
        ("What method joins elements of a list of strings with a specified delimiter?", "join", r"(?i)\bjoin\b"),
        ("What operator is used for matrix multiplication in Python 3.5+?", "@", r"@"),
        ("What built-in module provides regular expression matching operations?", "re", r"(?i)\bre\b"),
        ("What module provides mathematical functions such as sqrt, sin, and cos?", "math", r"(?i)\bmath\b"),
        ("What module provides platform-independent file and directory path operations?", "os.path or pathlib", r"(?i)(os|pathlib)"),
        ("What decorator is used to define a method bound to the class rather than instance?", "@classmethod", r"(?i)@?classmethod"),
        ("What decorator is used to define a method that behaves like an attribute getter?", "@property", r"(?i)@?property"),
        ("What method is called when an instance of a class is created in Python?", "__init__", r"__init__"),
        ("What dunder method defines string representation intended for end-users?", "__str__", r"__str__"),
        ("What dunder method defines unambiguous representation intended for developers/debugging?", "__repr__", r"__repr__"),
        ("What dunder method defines the length behavior when calling len() on an object?", "__len__", r"__len__"),
        ("What dunder method enables indexing syntax such as obj[key]?", "__getitem__", r"__getitem__"),
    ]
    for prompt, ans, reg in code_snippets:
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "python_syntax",
            "difficulty": "easy",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "python",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "python_reasoning",
            "difficulty": "medium",
            "prompt": f"Write a single Python expression or identify the exact keyword: {prompt}",
            "reference_answer": ans,
            "expected_action": "python",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 2. Time & Space Complexity (50)
    complexity_questions = [
        ("What is the average time complexity of searching an element in a balanced binary search tree?", "O(log n)", r"(?i)O\(\s*log\s*n\s*\)"),
        ("What is the average time complexity of lookups in a Python dictionary (hash table)?", "O(1)", r"(?i)O\(\s*1\s*\)"),
        ("What is the worst-case time complexity of standard Quicksort with naive pivot?", "O(n^2)", r"(?i)O\(\s*n(?:\^2|\*\*2)\s*\)"),
        ("What is the worst-case time complexity of Mergesort?", "O(n log n)", r"(?i)O\(\s*n\s*log\s*n\s*\)"),
        ("What is the time complexity of reversing a Python list of length n in-place?", "O(n)", r"(?i)O\(\s*n\s*\)"),
        ("What is the time complexity of accessing an array element by index arr[i]?", "O(1)", r"(?i)O\(\s*1\s*\)"),
        ("What is the worst-case time complexity of binary search on a sorted array of size n?", "O(log n)", r"(?i)O\(\s*log\s*n\s*\)"),
        ("What is the time complexity of Breadth-First Search (BFS) on a graph with V vertices and E edges?", "O(V + E)", r"(?i)O\(\s*V\s*\+\s*E\s*\)"),
        ("What is the time complexity of Depth-First Search (DFS) on a graph with V vertices and E edges?", "O(V + E)", r"(?i)O\(\s*V\s*\+\s*E\s*\)"),
        ("What is the time complexity of inserting an element at the beginning of a Python list (arr.insert(0, x))?", "O(n)", r"(?i)O\(\s*n\s*\)"),
        ("What is the amortized time complexity of appending an element to a Python list (arr.append(x))?", "O(1)", r"(?i)O\(\s*1\s*\)"),
        ("What data structure provides O(1) time complexity for both append and popleft operations in Python?", "collections.deque", r"(?i)deque"),
        ("What is the time complexity of computing Fibonacci recursively without memoization?", "O(2^n)", r"(?i)O\(\s*2\^n\s*\)"),
        ("What is the time complexity of computing Fibonacci iteratively in a loop?", "O(n)", r"(?i)O\(\s*n\s*\)"),
        ("What is the auxiliary space complexity of iterative binary search?", "O(1)", r"(?i)O\(\s*1\s*\)"),
        ("What is the space complexity of Mergesort on an array of size n?", "O(n)", r"(?i)O\(\s*n\s*\)"),
        ("What is the average time complexity of inserting into a min-heap or priority queue of size n?", "O(log n)", r"(?i)O\(\s*log\s*n\s*\)"),
        ("What algorithm finds the shortest path between nodes in a graph with non-negative edge weights?", "Dijkstra's algorithm", r"(?i)Dijkstra"),
        ("What algorithm detects negative-weight cycles in a weighted directed graph?", "Bellman-Ford algorithm", r"(?i)Bellman-Ford"),
        ("What data structure is used to implement a LIFO (Last-In-First-Out) order?", "Stack", r"(?i)stack"),
        ("What data structure is used to implement a FIFO (First-In-First-Out) order?", "Queue", r"(?i)queue"),
        ("What data structure is used to detect cycles in an undirected graph via disjoint sets?", "Union-Find (Disjoint Set Union / DSU)", r"(?i)(union-find|dsu|disjoint)"),
        ("What is the time complexity of matrix multiplication using the naive three-nested-loops algorithm for n x n matrices?", "O(n^3)", r"(?i)O\(\s*n(?:\^3|\*\*3)\s*\)"),
        ("What sorting algorithm is guaranteed to be stable and runs in O(n log n) time in all cases?", "Mergesort / Timsort", r"(?i)(merge\s*sort|timsort)"),
        ("What is the best-case time complexity of insertion sort on an already sorted array?", "O(n)", r"(?i)O\(\s*n\s*\)"),
    ]
    for prompt, ans, reg in complexity_questions:
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "algorithms_complexity",
            "difficulty": "medium",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "complexity_analysis",
            "difficulty": "medium",
            "prompt": f"Analyze complexity: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 3. Bug Hunting & Code Output Simulation (50)
    for i in range(25):
        val = i * 4 + 3
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "code_execution_simulation",
            "difficulty": "medium",
            "prompt": f"What is the exact output of this Python code:\ndef f(x):\n    return x * 2 + 1\nprint(f({val}))",
            "reference_answer": str(val * 2 + 1),
            "expected_action": "python",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{val * 2 + 1}\b"
        })
        n_items = 4 + (i % 6)
        sq_sum = sum(x**2 for x in range(n_items))
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "code_execution_simulation",
            "difficulty": "medium",
            "prompt": f"What is the exact output of: `print(sum(x**2 for x in range({n_items})))`?",
            "reference_answer": str(sq_sum),
            "expected_action": "python",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{sq_sum}\b"
        })

    # 4. Regex, JSON & Data Structures (50)
    regex_prompts = [
        ("In regex, what character matches any single character except newline?", ".", r"\."),
        ("In regex, what quantifier matches 0 or more occurrences?", "*", r"\*"),
        ("In regex, what quantifier matches 1 or more occurrences?", "+", r"\+"),
        ("In regex, what quantifier matches 0 or 1 occurrence?", "?", r"\?"),
        ("In regex, what anchor matches the beginning of a string?", "^", r"\^"),
        ("In regex, what anchor matches the end of a string?", "$", r"\$"),
        ("In regex, what shorthand character class matches any digit 0-9?", r"\d", r"\\d"),
        ("In regex, what shorthand character class matches any whitespace character?", r"\s", r"\\s"),
        ("In regex, what shorthand character class matches any word character (alphanumeric and underscore)?", r"\w", r"\\w"),
        ("What format is standard for exchanging structured key-value data across web APIs?", "JSON", r"(?i)\bJSON\b"),
        ("In JSON, can object keys be unquoted numbers or must they be double-quoted strings?", "Must be strings", r"(?i)string"),
        ("In Python, what function serializes a Python dictionary into a JSON formatted string?", "json.dumps", r"(?i)dumps"),
        ("In Python, what function deserializes a JSON string into a Python object?", "json.loads", r"(?i)loads"),
        ("What SQL keyword is used to retrieve unique records by eliminating duplicates?", "DISTINCT", r"(?i)DISTINCT"),
        ("What SQL clause filters rows grouped by an aggregation function?", "HAVING", r"(?i)HAVING"),
        ("What SQL clause filters individual rows before grouping?", "WHERE", r"(?i)WHERE"),
        ("What SQL command removes all rows from a table without logging individual row deletions?", "TRUNCATE", r"(?i)TRUNCATE"),
        ("What SQL constraint ensures that all values in a column are distinct?", "UNIQUE", r"(?i)UNIQUE"),
        ("What SQL key uniquely identifies each record in a database table?", "PRIMARY KEY", r"(?i)PRIMARY KEY"),
        ("What SQL key in one table references the primary key in another table?", "FOREIGN KEY", r"(?i)FOREIGN KEY"),
        ("What Git command displays the state of the working directory and staging area?", "git status", r"(?i)status"),
        ("What Git command records staged changes to the repository?", "git commit", r"(?i)commit"),
        ("What Linux command creates a new directory?", "mkdir", r"(?i)\bmkdir\b"),
        ("What Linux command displays the current working directory path?", "pwd", r"(?i)\bpwd\b"),
        ("What Linux command outputs the contents of a file to stdout?", "cat", r"(?i)\bcat\b"),
    ]
    for prompt, ans, reg in regex_prompts:
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "tooling_and_regex",
            "difficulty": "easy",
            "prompt": prompt,
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })
        items.append({
            "category": "code_and_algorithms",
            "sub_category": "tooling_quiz",
            "difficulty": "easy",
            "prompt": f"Identify the command or symbol: {prompt}",
            "reference_answer": ans,
            "expected_action": "chat",
            "eval_metric": "regex",
            "verification_regex": reg
        })

    return items[:200]


def build_needle_and_agentic_pillar():
    """Generates 200 needle-in-a-haystack sprinkled retrieval and agentic tool workflow prompts."""
    items = []

    # 1. Needle-in-a-Haystack Sprinkled Retrieval (60)
    first_names = ["Arthur", "Beatrice", "Cedric", "Diana", "Eleanor", "Felix", "Genevieve", "Hector", "Irene", "Jasper"]
    cities = ["Zurich", "Kyoto", "Valparaiso", "Reykjavik", "Nairobi", "Wellington", "Edinburgh", "Seville", "Calgary", "Innsbruck"]
    fruits = ["pomegranate", "kumquat", "dragonfruit", "persimmon", "fig", "boysenberry", "papaya", "guava", "lychee", "starfruit"]
    passcodes = [1048, 2941, 5823, 7192, 4038, 9512, 6384, 8271, 3605, 5149]

    for i in range(30):
        name = first_names[i % len(first_names)]
        city = cities[i % len(cities)]
        fruit = fruits[i % len(fruits)]
        code = passcodes[i % len(passcodes)]
        offset = i * 7

        haystack = (
            f"The annual summit convened early in the morning to discuss international logistics. "
            f"Delegates from various nations reviewed tariff agreements and trade documentation. "
            f"Special operative {name} was assigned to travel to {city} and securely deliver the secret code {code}. "
            f"During the afternoon session, participants enjoyed catered refreshments including fresh {fruit} and green tea. "
            f"The final committee vote was adjourned until next Tuesday morning."
        )
        # Needle 1: Secret code
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "needle_retrieval",
            "difficulty": "medium",
            "prompt": f"Read the passage carefully and extract the requested detail:\n\"{haystack}\"\nQuestion: What was the secret code assigned to {name}?",
            "reference_answer": str(code),
            "expected_action": "chat",
            "eval_metric": "numeric_match",
            "verification_regex": rf"\b{code}\b"
        })
        # Needle 2: Target City
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "needle_retrieval",
            "difficulty": "medium",
            "prompt": f"Read the passage carefully:\n\"{haystack}\"\nQuestion: Which city was {name} instructed to travel to?",
            "reference_answer": city,
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"(?i)\b{city}\b"
        })

    # 2. Agentic Tool Classification (80)
    # The model must classify the correct tool action: CHAT, BASH, PYTHON, FILE_READ, CALCULATOR
    tool_tasks = [
        ("Compute 142857 * 7 using an exact arithmetic tool.", "calculator", "calculator", r"(?i)(1000000|999999)"),
        ("What is 849204 divided by 12?", "calculator", "calculator", r"\b70767\b"),
        ("Evaluate (45 * 82) + (19 * 33).", "calculator", "calculator", r"\b4317\b"),
        ("Calculate 2 to the power of 16.", "calculator", "calculator", r"\b65536\b"),
        ("Compute the square root of 50625.", "calculator", "calculator", r"\b225\b"),
        ("List all files in the current workspace directory using a shell terminal.", "bash", "bash", r"(?i)bash|ls"),
        ("Check the disk usage of the current partition using bash.", "bash", "bash", r"(?i)df|bash"),
        ("Find all Python files in the repository using a shell command.", "bash", "bash", r"(?i)find|bash"),
        ("Check Git status and branch name using the terminal command.", "bash", "bash", r"(?i)git status|bash"),
        ("Print the current system date and UTC time via bash.", "bash", "bash", r"(?i)date|bash"),
        ("Run a Python script to compute the 20th Fibonacci number.", "python", "python", r"\b6765\b"),
        ("Execute Python code to sort a list of dictionary items by key.", "python", "python", r"(?i)sorted|python"),
        ("Simulate 10,000 coin tosses in Python to estimate the probability of getting 5 heads in a row.", "python", "python", r"(?i)random|python"),
        ("Write a Python one-liner to generate a list of all prime numbers under 50.", "python", "python", r"(?i)prime|python"),
        ("Execute a Python snippet using the datetime module to calculate days between two dates.", "python", "python", r"(?i)datetime|python"),
        ("Read the contents of the README.md file in the workspace.", "file_read", "file_read", r"(?i)README|file_read"),
        ("Inspect line 1 to 20 of requirements.txt.", "file_read", "file_read", r"(?i)requirements|file_read"),
        ("Open and view configs/model_config.yaml.", "file_read", "file_read", r"(?i)config|file_read"),
        ("Read the license information from LICENSE.txt.", "file_read", "file_read", r"(?i)license|file_read"),
        ("Examine the head of data/train.csv.", "file_read", "file_read", r"(?i)train\.csv|file_read"),
        ("Calculate 999 * 999 using the calculator.", "calculator", "calculator", r"\b998001\b"),
        ("Divide 1000000 by 125.", "calculator", "calculator", r"\b8000\b"),
        ("Compute 3.14159 * 100.", "calculator", "calculator", r"\b314\.159\b"),
        ("Evaluate 15 * 15 * 15.", "calculator", "calculator", r"\b3375\b"),
        ("Compute 2 ** 10.", "calculator", "calculator", r"\b1024\b"),
        ("Run `uname -a` to get operating system kernel details.", "bash", "bash", r"(?i)uname|bash"),
        ("Check available memory in megabytes using `free -m`.", "bash", "bash", r"(?i)free|bash"),
        ("List running processes using `ps aux`.", "bash", "bash", r"(?i)ps|bash"),
        ("Count lines of code in src directory using `wc -l`.", "bash", "bash", r"(?i)wc|bash"),
        ("Search for 'TODO' comments in code using `grep -rn`.", "bash", "bash", r"(?i)grep|bash"),
        ("Write a Python script to check if a word is an anagram of another.", "python", "python", r"(?i)sorted|python"),
        ("Write a Python generator function to produce infinite even numbers.", "python", "python", r"(?i)yield|python"),
        ("Compute the greatest common divisor of 54 and 24 using Python math.gcd.", "python", "python", r"\b6\b"),
        ("Parse an ISO timestamp string into a Python datetime object.", "python", "python", r"(?i)fromisoformat|python"),
        ("Encode a string into base64 using Python base64 module.", "python", "python", r"(?i)b64encode|python"),
        ("Read the environment variable file .env.", "file_read", "file_read", r"(?i)\.env|file_read"),
        ("Read the gitignore file to see ignored artifacts.", "file_read", "file_read", r"(?i)gitignore|file_read"),
        ("Read the first 50 lines of src/spheroid_v5.py.", "file_read", "file_read", r"(?i)spheroid_v5|file_read"),
        ("Read setup.py to verify package dependencies.", "file_read", "file_read", r"(?i)setup\.py|file_read"),
        ("Read docker-compose.yml to see active container services.", "file_read", "file_read", r"(?i)docker|file_read"),
    ]
    for prompt, expected_tool, ref, reg in tool_tasks:
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "tool_dispatch",
            "difficulty": "medium",
            "prompt": f"System Task: '{prompt}' -> Which specialized agentic action should be dispatched: CHAT, BASH, PYTHON, FILE_READ, or CALCULATOR?",
            "reference_answer": expected_tool,
            "expected_action": expected_tool,
            "eval_metric": "action_match",
            "verification_regex": rf"(?i)\b{expected_tool}\b"
        })
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "agentic_execution",
            "difficulty": "hard",
            "prompt": f"You are an autonomous AI agent. Given the user request: '{prompt}', produce the tool invocation or answer.",
            "reference_answer": ref,
            "expected_action": expected_tool,
            "eval_metric": "regex",
            "verification_regex": reg
        })

    # 3. Multi-Step Instruction Following & Constraints (60)
    for i in range(30):
        num_a = 15 + i * 2
        num_b = 8 + (i % 5)
        expected_sum = num_a + num_b
        expected_diff = num_a - num_b
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "multi_constraint",
            "difficulty": "medium",
            "prompt": f"Strict Instruction Following:\n1. Add {num_a} and {num_b}.\n2. Subtract {num_b} from {num_a}.\n3. Output ONLY the two results separated by a comma (Sum, Difference) with NO other text.",
            "reference_answer": f"{expected_sum}, {expected_diff}",
            "expected_action": "calculator",
            "eval_metric": "regex",
            "verification_regex": rf"\b{expected_sum}\s*,\s*{expected_diff}\b"
        })
        city_a = cities[i % len(cities)]
        city_b = cities[(i + 3) % len(cities)]
        items.append({
            "category": "needle_and_agentic",
            "sub_category": "multi_constraint_format",
            "difficulty": "medium",
            "prompt": f"Format constraint: Provide the first three letters of '{city_a}' and '{city_b}' in uppercase separated by a hyphen (e.g., ABC-DEF).",
            "reference_answer": f"{city_a[:3].upper()}-{city_b[:3].upper()}",
            "expected_action": "chat",
            "eval_metric": "exact_match",
            "verification_regex": rf"\b{city_a[:3].upper()}-{city_b[:3].upper()}\b"
        })

    return items[:200]


def main():
    print("=" * 70)
    print("🚀 Compiling Cerebro-1000 Benchmark Dataset...")
    print("=" * 70)

    p1 = build_linguistic_pillar()
    p2 = build_factual_pillar()
    p3 = build_math_logic_pillar()
    p4 = build_code_pillar()
    p5 = build_needle_and_agentic_pillar()

    all_prompts = []
    pillars = [
        ("linguistic_and_syntax", p1),
        ("factual_and_counterfactual", p2),
        ("mathematical_and_logic", p3),
        ("code_and_algorithms", p4),
        ("needle_and_agentic", p5),
    ]

    total_idx = 0
    for name, pillar_items in pillars:
        print(f"  • Pillar '{name}': {len(pillar_items)} prompts compiled.")
        for item in pillar_items:
            total_idx += 1
            item["id"] = f"cerebro_1000_{total_idx:04d}"
            all_prompts.append(item)

    print(f"\nTotal Prompts Assembled: {len(all_prompts)}")

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        for prompt_obj in all_prompts:
            f.write(json.dumps(prompt_obj) + "\n")

    print(f"✓ Cerebro-1000 Benchmark saved successfully to: {BENCHMARK_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
