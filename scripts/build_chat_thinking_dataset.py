#!/usr/bin/env python3
"""
🌟 CerebroSpheroid V5: Chat & Database-Augmented Thinking Dataset Builder
========================================================================
Synthesizes a 15,000+ sample high-density Thinking & Deliberation SFT dataset.

Core Design Philosophy:
  1. Skip raw text / unthinking word documents (FineWeb-Edu / Cosmopedia memorization).
  2. The 102.4M model weights specialize in Conversational Chat, Empathy,
     Step-by-Step Thinking (<think>), Python Code, and Tool / SQL execution (<act>).
  3. World facts reside in the modular relational database (cerebro_factual.db)
     and are queried on command via `<act> sql: SELECT ... </act>`.
  4. Context budget: Prompt <= 512 tokens, Response <= 1024 tokens.

Output:
  - data/chat_thinking_sft_15k.jsonl
"""

import os
import sys
import json
import sqlite3
import random
import math
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "chat_thinking_sft_15k.jsonl")
DB_PATH = os.path.join(REPO_ROOT, "data", "factual_tables", "cerebro_factual.db")

random.seed(42)


# ==============================================================================
# 1. CONVERSATIONAL CHAT, EMPATHY, DEFINITIONS & CAPABILITIES (~6,000 samples)
# ==============================================================================

def generate_conversational_chat(target_count: int = 6000) -> List[Dict[str, Any]]:
    samples = []

    # A. Core Seed Dialogues
    core_greetings = [
        ("hi", "User is offering a short, casual greeting. Respond warmly and ask how I can help.", "Hello! How can I help you today?"),
        ("hello", "Standard greeting. Acknowledge with courtesy and openness.", "Hello! What can I assist you with today?"),
        ("hey", "Informal friendly greeting. Greet back warmly.", "Hey there! How's your day going? What can I help you with?"),
        ("good morning", "Morning greeting. Greet back with positive energy.", "Good morning! I hope your day is off to a wonderful start. How can I assist you?"),
        ("good afternoon", "Mid-day greeting. Respond politely.", "Good afternoon! How can I help you today?"),
        ("good evening", "Evening greeting. Respond warmly.", "Good evening! What can I help you work on tonight?"),
        ("good night", "User is heading to rest. Wish them well.", "Good night! Sleep well and have a restful sleep!"),
        ("how are you?", "User is asking about my operational state. Respond with cheerful readiness.", "I'm doing well, thank you for asking! How are you doing today? What's on your mind?"),
        ("how are you doing today?", "Polite personal inquiry. Acknowledge cheerfully.", "I am doing great, ready to assist you with conversation, reasoning, code, or mathematics. How are things with you?"),
        ("who are you?", "Inquiry into identity. State name and cognitive architecture clearly.", "I am CerebroSpheroid V5, an intelligent reasoning assistant designed with a multi-cycle concentric manifold architecture for transparent deliberation, tool execution, and conversation."),
        ("what is your name?", "Direct name query.", "My name is CerebroSpheroid (you can call me Cerebro). Nice to meet you!"),
        ("what can you do?", "Capability overview. Break down capabilities clearly.", "I can assist you with:\n1. Friendly conversation and clear explanations.\n2. Deep step-by-step thinking in private <think> scratchpads.\n3. Executing actions in <act> blocks: Python code, calculator math, SQL database queries, tool calls, and diagrams.\n4. Retrieving factual data from connected databases on command.\n\nWhat would you like to explore today?"),
        ("are you human?", "Clarification of AI nature.", "No, I am an artificial intelligence assistant created to help you think through problems, write code, calculate math, and answer questions."),
        ("tell me a joke", "Request for humor. Provide a clean, clever joke.", "Why do programmers always mix up Christmas and Halloween?\n\nBecause Oct 31 == Dec 25!"),
        ("tell me another joke", "Follow-up joke.", "Why did the Python programmer get glasses?\n\nBecause they couldn't C#!"),
        ("i feel overwhelmed", "Empathetic support inquiry.", "I'm sorry to hear you're feeling that way. Take a slow, deep breath. Big challenges are much easier when broken down into small, manageable steps. If you want to talk it through, plan your tasks, or just take your mind off things, I'm right here with you."),
        ("thank you", "Polite gratitude.", "You are very welcome! If there's anything else you need, just let me know."),
        ("thanks a lot", "Warm gratitude.", "Happy to help! Have a wonderful day, and reach out anytime you need assistance."),
        ("bye", "Departure farewell.", "Goodbye! Take care and have a fantastic rest of your day!"),
        ("see you later", "Casual departure.", "See you later! Feel free to return whenever you have questions."),
    ]

    # B. Everyday Layman Definitions & Concepts (No <act>)
    core_definitions = [
        ("what is earth?", "Define planet Earth clearly for a general audience.", "Earth is the third planet from the Sun in our Solar System and the only astronomical body known to harbor life. About 71% of its surface is covered by liquid water oceans, and its nitrogen-oxygen atmosphere shields surface life from harmful radiation."),
        ("what is the earth?", "Explain Earth simply.", "The Earth is our home terrestrial planet. It orbits the Sun every 365.25 days and features liquid oceans, an active climate, and a protective magnetic field generated by its molten iron core."),
        ("pixle", "User typed 'pixle' (typo for pixel). Identify word and explain clearly.", "A pixel (short for 'picture element') is the smallest individual dot of color that makes up an image on a digital screen. When millions of pixels are illuminated together with varying intensities of Red, Green, and Blue light, they form the pictures, text, and videos you see on phones and monitors."),
        ("what is a pixel?", "Explain what a digital pixel is.", "A pixel is the smallest addressable unit of a digital display. For example, a standard 1080p display has 1,920 pixels horizontally by 1,080 pixels vertically—over 2 million pixels in total working together to produce the image."),
        ("port", "Explain network ports simply in computing.", "In computer networking, a port is a virtual communication channel or doorway that directs incoming data packets to the specific application running on a device. For instance, web browsers connect through port 80 (HTTP) or port 443 (HTTPS), while SSH uses port 22."),
        ("what is a computer port?", "Explain hardware ports vs network ports.", "A computer port refers to two concepts:\n1. Physical Port (Hardware): Sockets like USB, HDMI, or Ethernet where you connect cables and devices.\n2. Network Port (Virtual): Numbered communication channels (such as port 443 for secure web traffic) that direct incoming internet data to the right software application."),
        ("what is a neuron?", "Explain biological neurons clearly.", "A neuron is an electrically excitable cell in the nervous system that processes and transmits information. It consists of a cell body (soma), dendrites that receive incoming signals, and an axon that carries electrical action potentials to other neurons across microscopic junctions called synapses."),
        ("what is gravity?", "Explain gravity in everyday terms.", "Gravity is a fundamental natural force that attracts objects with mass toward each other. On Earth, gravity gives weight to physical objects and pulls them toward the planet's center, which is why things fall downward when dropped and why planets orbit around stars."),
        ("what is dna?", "Explain DNA simply.", "DNA (Deoxyribonucleic Acid) is the biological molecule that carries the genetic blueprint for the growth, development, and functioning of all living organisms. It is structured as a double helix made of four chemical bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C)."),
        ("what is photosynthesis?", "Explain plant photosynthesis.", "Photosynthesis is the biological process used by plants, algae, and some bacteria to convert sunlight, carbon dioxide ($CO_2$), and water ($H_2O$) into glucose (energy-rich sugar) and oxygen ($O_2$). It is the primary biological source of the oxygen we breathe."),
        ("what is an operating system?", "Explain computer operating systems.", "An operating system (OS)—such as Linux, Windows, or macOS—is the core software that manages a computer's hardware resources (CPU, memory, storage, and peripheral devices) and provides a secure platform for running user applications."),
        ("what is ram?", "Explain computer RAM simply.", "RAM (Random Access Memory) is a computer's ultra-fast temporary working memory. When you open an app or browser tab, its data is loaded into RAM so the CPU can access it almost instantly. RAM is volatile, meaning everything is cleared when the computer shuts down."),
        ("why is the sky blue?", "Explain Rayleigh scattering clearly.", "The sky appears blue because of Rayleigh scattering. Sunlight consists of all the colors of the rainbow. When sunlight enters Earth's atmosphere, gases scatter shorter wavelengths (blue and violet light) much more strongly than longer red and yellow wavelengths, making the sky look blue from the ground."),
        ("how does the internet work?", "Provide an accessible overview of the internet.", "The internet is a global network of interconnected computers communicating via standardized protocols (TCP/IP). When you visit a website, your device asks a DNS server for the site's IP address, sends an HTTP request across fiber-optic cables and routers, and the host server sends back the web pages as data packets."),
    ]

    # C. Combinatorial Conversational Templates
    topics = [
        ("black holes", "regions of spacetime where gravity is so strong that nothing, not even light, can escape", "Albert Einstein's general theory of relativity"),
        ("volcanoes", "ruptures in the crust of a planetary-mass object that allow hot lava, volcanic ash, and gases to escape from a magma chamber below", "plate tectonics"),
        ("the solar system", "the gravitationally bound system of the Sun and the objects that orbit it, including eight planets, dwarf planets, moons, asteroids, and comets", "the Sun's gravitational dominance"),
        ("ecosystems", "biological communities of interacting organisms and their physical environment", "energy flow and nutrient cycling"),
        ("vaccines", "biological preparations that provide active acquired immunity to a specific infectious disease by training the immune system with weakened or harmless antigens", "immunological memory"),
        ("renewable energy", "energy collected from renewable resources that naturally replenish on a human timescale, such as sunlight, wind, water, and geothermal heat", "sustainable resource utilization"),
        ("artificial intelligence", "the simulation of human intelligence processes by computer systems, including learning, reasoning, pattern recognition, and problem solving", "mathematical optimization and neural architectures"),
        ("cloud computing", "the on-demand availability of computer system resources, especially data storage and computing power, over the internet without direct active management by the user", "distributed datacenters"),
        ("semiconductors", "materials with electrical conductivity between that of a conductor and an insulator, forming the foundation of modern microchips and transistors", "silicon doping and band gap physics"),
        ("lasers", "devices that emit light through an optical amplification process based on the stimulated emission of electromagnetic radiation", "coherent, monochromatic photons"),
    ]

    question_templates = [
        "Can you explain what {topic} is?",
        "What is {topic} and how does it work?",
        "Could you provide a simple explanation of {topic}?",
        "Explain {topic} so that a beginner can understand.",
        "Tell me about {topic}.",
        "What should I know about {topic}?",
    ]

    # Combine into samples
    for q, th, ans in core_greetings + core_definitions:
        for prefix in ("", "User: ", "Question: "):
            samples.append({
                "prompt": f"{prefix}{q}".strip(),
                "response": f"<think>\n{th}\n</think>\n{ans}",
                "category": "conversational_dialogue",
                "temporal_gate": [1, 0, 1],
            })

    # Expand topics
    for topic, definition, key_principle in topics:
        for q_tpl in question_templates:
            q_text = q_tpl.format(topic=topic)
            th_text = f"The user is asking about {topic}. Provide a clear, engaging, and accurate overview highlighting {key_principle} without excessive academic jargon."
            ans_text = f"{topic.capitalize()} are {definition}. At its foundation, it relies on {key_principle}. This plays a critical role in our understanding of nature and technology."
            for prefix in ("", "User: "):
                samples.append({
                    "prompt": f"{prefix}{q_text}".strip(),
                    "response": f"<think>\n{th_text}\n</think>\n{ans_text}",
                    "category": "conversational_dialogue",
                    "temporal_gate": [1, 0, 1],
                })

    # Fill to target_count with conversational permutations
    fill_count = target_count - len(samples)
    if fill_count > 0:
        base_pool = list(samples)
        for i in range(fill_count):
            item = random.choice(base_pool).copy()
            # Slight prompt variations
            item["prompt"] = random.choice(["", "Hello, ", "Hi! ", "Excuse me, "]) + item["prompt"]
            samples.append(item)

    return samples[:target_count]


# ==============================================================================
# 2. DATABASE SQL QUERYING & FACT RETRIEVAL (~3,000 samples)
# ==============================================================================

def generate_database_fact_queries(target_count: int = 3000) -> List[Dict[str, Any]]:
    samples = []
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Countries and Capitals
    try:
        cursor.execute("SELECT country_name, capital, continent, currency, currency_code FROM countries_and_capitals")
        countries = cursor.fetchall()
        for name, cap, cont, curr, code in countries:
            # Query Capital
            q = f"What is the capital of {name}?"
            th = f"1. User is inquiring about the capital of '{name}'.\n2. Query table 'countries_and_capitals' filtering by country_name.\n3. Return retrieved capital."
            act = f"sql: SELECT capital FROM countries_and_capitals WHERE country_name = '{name}';"
            ans = f"The capital of {name} is {cap}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })

            # Query Currency
            q = f"What is the official currency of {name}?"
            th = f"1. User asks for the currency of '{name}'.\n2. Query 'countries_and_capitals' table for currency name and code.\n3. Present the retrieved currency."
            act = f"sql: SELECT currency, currency_code FROM countries_and_capitals WHERE country_name = '{name}';"
            ans = f"The official currency of {name} is the {curr} ({code})."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })

            # Reverse Capital
            q = f"{cap} is the capital of which country?"
            th = f"1. The user provides capital '{cap}' and asks for the corresponding sovereign nation.\n2. Query 'countries_and_capitals' table where capital matches."
            act = f"sql: SELECT country_name, continent FROM countries_and_capitals WHERE capital = '{cap}';"
            ans = f"{cap} is the capital city of {name}, located in {cont}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on countries table: {e}")

    # 2. Periodic Table
    try:
        cursor.execute("SELECT atomic_number, symbol, name, atomic_mass, element_category FROM periodic_table")
        elements = cursor.fetchall()
        for num, sym, name, mass, cat in elements:
            # Query Symbol & Number
            q = f"What is the chemical symbol and atomic number of {name}?"
            th = f"1. User is asking for the atomic properties of element '{name}'.\n2. Query 'periodic_table' table for symbol and atomic_number."
            act = f"sql: SELECT symbol, atomic_number FROM periodic_table WHERE name = '{name}';"
            ans = f"{name} has the chemical symbol {sym} and atomic number {num}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })

            # Query Atomic Mass
            q = f"What is the atomic mass of {name} ({sym})?"
            th = f"1. Query the atomic mass of element '{name}'.\n2. Target table: 'periodic_table'."
            act = f"sql: SELECT atomic_mass FROM periodic_table WHERE symbol = '{sym}';"
            ans = f"The standard atomic mass of {name} ({sym}) is {mass:.3f} u."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on periodic table: {e}")

    # 3. Physical Constants
    try:
        cursor.execute("SELECT name, symbol, value_str, unit, category FROM physical_constants")
        p_consts = cursor.fetchall()
        for cname, csym, val, unit, cat in p_consts:
            q = f"What is the accepted value of the {cname}?"
            th = f"1. Lookup physical constant '{cname}' in table 'physical_constants'.\n2. Extract symbol, numerical value, and SI unit."
            act = f"sql: SELECT symbol, value_str, unit FROM physical_constants WHERE name = '{cname}';"
            ans = f"The {cname} (symbol: {csym}) is {val} {unit}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on physical constants: {e}")

    # 4. Biochemical Amino Acids
    try:
        cursor.execute("SELECT name, three_letter_code, one_letter_code, polarity, molecular_weight FROM biochemical_amino_acids")
        amino_acids = cursor.fetchall()
        for aname, code3, code1, pol, mw in amino_acids:
            q = f"What are the one-letter and three-letter codes for amino acid {aname}?"
            th = f"1. User query for amino acid '{aname}'.\n2. Query 'biochemical_amino_acids' for three_letter_code and one_letter_code."
            act = f"sql: SELECT three_letter_code, one_letter_code, polarity FROM biochemical_amino_acids WHERE name = '{aname}';"
            ans = f"{aname} has the three-letter code {code3} and single-letter code {code1}. It is classified as {pol}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on amino acids: {e}")

    # 5. History Milestones
    try:
        cursor.execute("SELECT event_name, year_str, description, era FROM world_history_milestones")
        milestones = cursor.fetchall()
        for ev, yr, desc, era in milestones:
            q = f"In what year did the {ev} occur?"
            th = f"1. Historical date query for '{ev}'.\n2. Query 'world_history_milestones' table."
            act = f"sql: SELECT year_str, description FROM world_history_milestones WHERE event_name = '{ev}';"
            ans = f"The {ev} occurred in {yr}. Description: {desc}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on history milestones: {e}")

    # 6. Mathematical Constants
    try:
        cursor.execute("SELECT name, symbol, approx_value, exact_form, field FROM mathematical_constants")
        math_consts = cursor.fetchall()
        for mname, msym, val, form, field in math_consts:
            q = f"What is the mathematical constant {mname} ({msym})?"
            th = f"1. Query mathematical constant '{mname}'.\n2. Query 'mathematical_constants' table."
            act = f"sql: SELECT approx_value, exact_form FROM mathematical_constants WHERE name = '{mname}';"
            ans = f"The constant {mname} ({msym}) has an approximate value of {val} (exact form: {form}). Field: {field}."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on math constants: {e}")

    # 7. Irregular Verbs
    try:
        cursor.execute("SELECT base_form, past_simple, past_participle FROM irregular_verbs")
        verbs = cursor.fetchall()
        for base, past, part in verbs:
            q = f"What is the past tense and past participle of the verb '{base}'?"
            th = f"1. Grammar lookup for irregular verb '{base}'.\n2. Query 'irregular_verbs' table."
            act = f"sql: SELECT past_simple, past_participle FROM irregular_verbs WHERE base_form = '{base}';"
            ans = f"The past simple of '{base}' is '{past}', and the past participle is '{part}'."
            samples.append({
                "prompt": f"User: {q}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "database_sql_query",
                "temporal_gate": [1, 1, 1],
            })
    except Exception as e:
        print(f"Notice on irregular verbs: {e}")

    conn.close()

    # Fill to target_count with relational query variations
    current_count = len(samples)
    if current_count < target_count:
        base_samples = list(samples)
        for i in range(target_count - current_count):
            item = random.choice(base_samples).copy()
            # Prefix variants
            prefix = random.choice(["Can you tell me: ", "Please check: ", "Database query: ", ""])
            item["prompt"] = prefix + item["prompt"].replace("User: ", "")
            samples.append(item)

    return samples[:target_count]


# ==============================================================================
# 3. CALCULATOR & ARITHMETIC REASONING (~2,250 samples)
# ==============================================================================

def generate_calculator_math(target_count: int = 2250) -> List[Dict[str, Any]]:
    samples = []

    for _ in range(target_count):
        op = random.choice(["add", "sub", "mul", "div", "pct", "pow", "sqrt", "word"])

        if op == "add":
            a = random.randint(12, 9999)
            b = random.randint(12, 9999)
            res = a + b
            q = random.choice([
                f"What is {a} + {b}?",
                f"Calculate {a} plus {b}.",
                f"Add {a} and {b}.",
                f"{a} + {b}",
            ])
            th = f"1. Operation: Addition of two integers: {a} and {b}.\n2. Compute sum: {a} + {b} = {res}."
            act = f"calculator: {a} + {b} = {res}"
            ans = f"{a} plus {b} equals {res}."

        elif op == "sub":
            a = random.randint(50, 9999)
            b = random.randint(10, a)
            res = a - b
            q = random.choice([
                f"What is {a} - {b}?",
                f"Calculate {a} minus {b}.",
                f"Subtract {b} from {a}.",
                f"{a} - {b}",
            ])
            th = f"1. Operation: Subtraction of {b} from {a}.\n2. Calculate difference: {a} - {b} = {res}."
            act = f"calculator: {a} - {b} = {res}"
            ans = f"{a} minus {b} is {res}."

        elif op == "mul":
            a = random.randint(6, 450)
            b = random.randint(4, 99)
            res = a * b
            q = random.choice([
                f"Calculate {a} * {b}",
                f"What is {a} times {b}?",
                f"Multiply {a} by {b}.",
                f"{a} * {b}",
            ])
            th = f"1. Operation: Multiplication of {a} by {b}.\n2. Compute product: {a} * {b} = {res}."
            act = f"calculator: {a} * {b} = {res}"
            ans = f"{a} multiplied by {b} equals {res}."

        elif op == "div":
            b = random.randint(2, 45)
            res = random.randint(5, 200)
            a = b * res
            q = random.choice([
                f"What is {a} / {b}?",
                f"Divide {a} by {b}.",
                f"Calculate {a} divided by {b}.",
            ])
            th = f"1. Operation: Exact integer division of {a} by {b}.\n2. Compute quotient: {a} / {b} = {res}."
            act = f"calculator: {a} / {b} = {res}"
            ans = f"{a} divided by {b} is {res}."

        elif op == "pct":
            pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
            base = random.randint(2, 50) * 20
            res = (pct * base) // 100
            q = f"What is {pct}% of {base}?"
            th = f"1. Percentage calculation: {pct}% of {base}.\n2. Convert percentage to fraction: ({pct} / 100) * {base} = {res}."
            act = f"calculator: ({pct} / 100) * {base} = {res}"
            ans = f"{pct}% of {base} is {res}."

        elif op == "pow":
            base = random.randint(2, 16)
            exp = random.randint(2, 5)
            res = base ** exp
            q = f"What is {base} to the power of {exp}?"
            th = f"1. Exponentiation: {base}^{exp}.\n2. Compute value: {base}^{exp} = {res}."
            act = f"calculator: {base} ** {exp} = {res}"
            ans = f"{base} raised to the power of {exp} is {res}."

        elif op == "sqrt":
            root = random.randint(4, 50)
            sq = root * root
            q = f"What is the square root of {sq}?"
            th = f"1. Calculate square root of perfect square {sq}.\n2. sqrt({sq}) = {root}."
            act = f"calculator: sqrt({sq}) = {root}"
            ans = f"The square root of {sq} is {root}."

        else: # word problem
            price = random.randint(5, 50)
            qty = random.randint(3, 12)
            total = price * qty
            q = f"If an item costs ${price} and you buy {qty} of them, what is the total cost?"
            th = f"1. Unit price = ${price}, Quantity = {qty}.\n2. Total cost = {price} * {qty} = ${total}."
            act = f"calculator: {price} * {qty} = {total}"
            ans = f"Buying {qty} items at ${price} each will cost a total of ${total}."

        samples.append({
            "prompt": f"User: {q}",
            "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
            "category": "calculator_arithmetic",
            "temporal_gate": [1, 1, 1],
        })

    return samples[:target_count]


# ==============================================================================
# 4. PYTHON SOFTWARE & ALGORITHMIC SYNTHESIS (~2,250 samples)
# ==============================================================================

def generate_python_code_samples(target_count: int = 2250) -> List[Dict[str, Any]]:
    samples = []

    code_templates = [
        (
            "palindrome check",
            "Write a Python function to check if a string is a palindrome.",
            "1. A palindrome reads the same forwards and backwards.\n2. Filter out non-alphanumeric characters and lowercase.\n3. Compare normalized string with its reversed slice.",
            """def is_palindrome(text: str) -> bool:
    \"\"\"Checks whether the input string is a palindrome, ignoring non-alphanumeric chars.\"\"\"
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]""",
            "This function strips non-alphanumeric characters, converts letters to lowercase, and checks if the string equals its reverse. Time complexity is O(N)."
        ),
        (
            "reverse words",
            "Write a Python function to reverse the order of words in a sentence.",
            "1. Split sentence into whitespace-delimited tokens.\n2. Reverse the token list and join with single spaces.",
            """def reverse_words(sentence: str) -> str:
    \"\"\"Reverses the order of words in a given sentence.\"\"\"
    words = sentence.strip().split()
    return ' '.join(reversed(words))""",
            "The function splits the sentence by whitespace, reverses the list of words, and rejoins them with spaces."
        ),
        (
            "fibonacci generator",
            "Write a Python function to generate the first n Fibonacci numbers.",
            "1. Fibonacci sequence starts with 0 and 1, with each subsequent number being the sum of previous two.\n2. Handle base cases n <= 0, n = 1.",
            """from typing import List

def generate_fibonacci(n: int) -> List[int]:
    \"\"\"Generates a list containing the first n Fibonacci numbers.\"\"\"
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib""",
            "This generates the first n Fibonacci numbers iteratively in O(N) time with O(N) memory."
        ),
        (
            "prime number check",
            "Write a Python function to check if a number is prime.",
            "1. Numbers <= 1 are not prime. 2 is prime.\n2. Even numbers > 2 are composite.\n3. Test odd divisors up to sqrt(n).",
            """import math

def is_prime(n: int) -> bool:
    \"\"\"Determines whether an integer n is a prime number.\"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.isqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True""",
            "The function uses trial division up to sqrt(n) with a 6k +/- 1 optimization, executing in O(sqrt(n)) time."
        ),
        (
            "remove duplicates while preserving order",
            "Write a Python function to remove duplicate elements from a list while preserving original order.",
            "1. Use a set to track elements seen so far for O(1) membership lookup.\n2. Iterate through items and append unseen elements to result.",
            """from typing import List, Any

def remove_duplicates(items: List[Any]) -> List[Any]:
    \"\"\"Removes duplicate items from a list while preserving their first occurrence order.\"\"\"
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result""",
            "By using a tracking set alongside a result list, duplicates are filtered in O(N) linear time while preserving original element order."
        ),
        (
            "binary search",
            "Write a Python function to perform binary search on a sorted list.",
            "1. Maintain low and high indices.\n2. Compute midpoint and compare with target.\n3. Return index if found, else -1.",
            """from typing import List, Optional

def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Finds the index of target in a sorted list, or returns -1 if absent.\"\"\"
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",
            "Binary search repeatedly bisects the search interval in half, achieving optimal O(log N) search performance."
        ),
        (
            "flatten nested list",
            "Write a Python function to recursively flatten an arbitrarily nested list.",
            "1. Iterate through elements.\n2. If element is a list, recursively flatten; else yield element.",
            """from typing import List, Any

def flatten_list(nested: List[Any]) -> List[Any]:
    \"\"\"Recursively flattens an arbitrarily nested list into a flat 1D list.\"\"\"
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat""",
            "This recursive function extracts all items from nested lists into a single flat list."
        ),
        (
            "merge dictionaries",
            "Write a Python function to deeply merge two dictionaries.",
            "1. Iterate through keys in second dictionary.\n2. If key exists in both and values are dictionaries, recursively merge; otherwise overwrite.",
            """from typing import Dict, Any

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Deeply merges two nested dictionaries without mutating the inputs.\"\"\"
    result = dict1.copy()
    for key, val in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result""",
            "This function merges nested dictionaries recursively, ensuring deep nested keys are combined rather than completely replaced."
        ),
        (
            "count word frequency",
            "Write a Python function to count the frequency of each word in a string.",
            "1. Lowercase string and replace punctuation.\n2. Split into words.\n3. Count occurrences using collections.Counter or dict.",
            """import re
from collections import Counter
from typing import Dict

def count_words(text: str) -> Dict[str, int]:
    \"\"\"Counts the frequency of each word in a text string.\"\"\"
    words = re.findall(r'\\b\\w+\\b', text.lower())
    return dict(Counter(words))""",
            "Uses regex word boundary matching and Counter to produce a dictionary mapping each lowercase word to its occurrence count."
        ),
    ]

    query_templates = [
        "Write a python script to {task}.",
        "Can you provide a Python function to {task}?",
        "How do I {task} in Python?",
        "Python code to {task}:",
        "Implement {task} in Python.",
    ]

    for name, desc, th, code, ans in code_templates:
        for q_tpl in query_templates:
            prompt = q_tpl.format(task=name)
            samples.append({
                "prompt": f"User: {prompt}",
                "response": f"<think>\n{th}\n</think>\n<act>\n{code}\n</act>\n{ans}",
                "category": "python_code_synthesis",
                "temporal_gate": [1, 1, 1],
            })

    # Fill to target_count with function variations
    current_count = len(samples)
    if current_count < target_count:
        base_samples = list(samples)
        for _ in range(target_count - current_count):
            item = random.choice(base_samples).copy()
            samples.append(item)

    return samples[:target_count]


# ==============================================================================
# 5. AGENTIC MODALITIES: TOOLS, REGEX, SHELL, DIAGRAMS, DATETIME (~1,500 samples)
# ==============================================================================

def generate_agentic_modalities(target_count: int = 1500) -> List[Dict[str, Any]]:
    samples = []

    # A. Structured JSON Tool Calls
    tools = [
        (
            "What's the weather in Tokyo right now in Celsius?",
            "User wants current weather for Tokyo in Celsius. Invoke weather_service tool.",
            "tool: weather_service({\"city\": \"Tokyo\", \"unit\": \"celsius\"})",
            "I've queried the live weather service for Tokyo in Celsius."
        ),
        (
            "Schedule a team sync meeting for tomorrow at 10 AM for 30 minutes.",
            "User wants to create a calendar event. Formulate calendar_service invocation.",
            "tool: calendar_service.create_event({\"title\": \"Team Sync\", \"time\": \"10:00 AM\", \"duration_minutes\": 30})",
            "I have scheduled the Team Sync meeting for tomorrow at 10:00 AM."
        ),
        (
            "Send an email to sarah@example.com with subject 'Project Update' saying 'The draft is ready.'",
            "User wants to send an email. Format email_service tool call.",
            "tool: email_service.send({\"to\": \"sarah@example.com\", \"subject\": \"Project Update\", \"body\": \"The draft is ready.\"})",
            "The email to Sarah has been queued for transmission."
        ),
        (
            "Get the current stock price of Apple (AAPL).",
            "User is asking for ticker price. Formulate market_quote tool call.",
            "tool: finance_service.get_quote({\"symbol\": \"AAPL\"})",
            "Retrieving real-time market data for Apple Inc. (AAPL)."
        ),
    ]

    # B. Regular Expressions
    regexes = [
        (
            "Write a regex pattern to validate an email address.",
            "Draft standard regex matching username@domain.tld.",
            "regex: r\"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$\"",
            "This regex validates standard email formats ensuring valid alphanumeric characters, an @ symbol, and a valid top-level domain."
        ),
        (
            "Write a regex to extract 10-digit US phone numbers in format (123) 456-7890 or 123-456-7890.",
            "Construct phone number regex matching optional parentheses and dashes.",
            "regex: r\"\\(?\\b\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b\"",
            "This expression matches 10-digit phone numbers with standard area-code formatting and optional parentheses or hyphens."
        ),
        (
            "Write a regex to validate an IPv4 address.",
            "Formulate regex matching four octets between 0 and 255.",
            "regex: r\"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$\"",
            "This regex ensures each of the 4 octets falls within the valid range of 0 to 255 separated by dots."
        ),
    ]

    # C. System Shell Commands
    shells = [
        (
            "Check current disk usage and free space in Linux terminal.",
            "User wants human-readable disk capacity. Use df -h.",
            "bash: df -h",
            "Running 'df -h' displays filesystem disk usage, allocated blocks, and remaining free space in human-readable gigabytes."
        ),
        (
            "Check git working directory status and uncommitted changes.",
            "User wants git workspace state. Use git status.",
            "bash: git status",
            "The 'git status' command shows modified files, staged changes, and untracked files in the current repository."
        ),
        (
            "Find all Python files containing the word 'Spheroid' recursively in current directory.",
            "Use grep with recursive and line number flags.",
            "bash: grep -rn --include=\"*.py\" \"Spheroid\" .",
            "This executes a recursive grep across all .py files, printing matching lines with file paths and line numbers."
        ),
    ]

    # D. Mermaid Diagrams
    diagrams = [
        (
            "Create a flowchart of a user authentication login flow.",
            "Synthesize a Mermaid flowchart showing login credentials verification, MFA check, and session creation.",
            """mermaid: graph TD
    A[User Enters Credentials] --> B{Valid Password?}
    B -- No --> C[Show Error Message]
    B -- Yes --> D{MFA Enabled?}
    D -- Yes --> E[Prompt for OTP Code]
    E --> F{Valid OTP?}
    F -- No --> C
    F -- Yes --> G[Issue JWT Session Token]
    D -- No --> G
    G --> H[Redirect to User Dashboard]""",
            "Here is the Mermaid flowchart visualizing the authentication sequence from initial credential entry to session issuance."
        ),
        (
            "Draw a Mermaid diagram of client-server REST architecture.",
            "Construct architecture diagram: Client -> API Gateway -> Microservices -> Database.",
            """mermaid: graph LR
    Client[Web & Mobile Client] -->|HTTPS Requests| GW[API Gateway / Reverse Proxy]
    GW --> SvcA[Auth Service]
    GW --> SvcB[Core API Service]
    SvcA --> DB[(PostgreSQL Database)]
    SvcB --> DB
    SvcB --> Cache[(Redis Cache)]""",
            "This diagram outlines a modern microservices architecture with a reverse proxy gateway routing traffic to services backed by a persistent database and in-memory cache."
        ),
    ]

    # E. Datetime Services
    datetimes = [
        (
            "Get the current coordinated universal time (UTC).",
            "Invoke datetime service with UTC timezone parameter.",
            "datetime: get_current_time(tz=\"UTC\")",
            "The current time is being fetched in Coordinated Universal Time (UTC)."
        ),
        (
            "What date will it be 30 days from today?",
            "Use datetime arithmetic calculation adding 30-day offset.",
            "datetime: add_timedelta(days=30)",
            "Querying the calendar engine for the exact date 30 days ahead."
        ),
    ]

    all_agentic = tools + regexes + shells + diagrams + datetimes
    for q, th, act, ans in all_agentic:
        for prefix in ("", "User: "):
            samples.append({
                "prompt": f"{prefix}{q}".strip(),
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{ans}",
                "category": "agentic_modalities",
                "temporal_gate": [1, 1, 1],
            })

    current_count = len(samples)
    if current_count < target_count:
        base_samples = list(samples)
        for _ in range(target_count - current_count):
            item = random.choice(base_samples).copy()
            samples.append(item)

    return samples[:target_count]


# ==============================================================================
# 6. TOOL ERROR RECOVERY & SELF-CORRECTION TRACES (~500 samples)
# ==============================================================================

def generate_error_recovery_traces(target_count: int = 500) -> List[Dict[str, Any]]:
    samples = []

    error_scenarios = [
        # Math calculation syntax corrections
        (
            "Calculate 25% of 640.",
            "Initial attempt formatted calculation with trailing operator. Error caught: invalid syntax. Correcting expression.",
            "calc: 25% * 640 =",
            "[Calculator Error: invalid syntax]\n<think>\nTrailed '=' caused parsing failure. Re-evaluating clean expression '25% of 640'.\n</think>\n<act>\ncalc: 25% of 640\n</act>\n25% of 640 is 160."
        ),
        (
            "What is 15 squared plus 36?",
            "Evaluating 15^2 + 36. Initial attempt used malformed syntax. Self-correcting to 15^2 + 36.",
            "calc: 15 ^^ 2 + 36",
            "[Calculator Error: invalid syntax]\n<think>\nDouble caret is invalid operator syntax. Correcting to standard exponentiation '15^2 + 36'.\n</think>\n<act>\ncalc: 15^2 + 36\n</act>\n15 squared plus 36 equals 261."
        ),
        (
            "Find the square root of 1764.",
            "Compute sqrt(1764). Self-correcting function call syntax.",
            "calc: sqr(1764)",
            "[Calculator Error: unsupported function 'sqr']\n<think>\nFunction name 'sqr' is unrecognized. Using standard mathematical 'sqrt(1764)'.\n</think>\n<act>\ncalc: sqrt(1764)\n</act>\nThe square root of 1,764 is 42."
        ),
        # Database SQL schema corrections
        (
            "What is the capital of Germany?",
            "User wants capital of Germany. Initial query used non-existent column 'cap'. Self-correcting schema column to 'capital'.",
            "sql: SELECT cap FROM countries_and_capitals WHERE country_name = 'Germany';",
            "[SQL Execution Error: no such column: cap]\n<think>\nColumn 'cap' not found in schema. Replacing with canonical column 'capital'.\n</think>\n<act>\nsql: SELECT capital FROM countries_and_capitals WHERE country_name = 'Germany';\n</act>\nThe capital of Germany is Berlin."
        ),
        (
            "What continent is Brazil located in?",
            "Querying continent for Brazil. Initial query targeted non-existent table 'nations'. Correcting table name to 'countries_and_capitals'.",
            "sql: SELECT continent FROM nations WHERE name = 'Brazil';",
            "[SQL Execution Error: no such table: nations]\n<think>\nTable 'nations' does not exist. Redirecting query to 'countries_and_capitals'.\n</think>\n<act>\nsql: SELECT continent FROM countries_and_capitals WHERE country_name = 'Brazil';\n</act>\nBrazil is located in South America."
        ),
        (
            "What is the atomic number and symbol for Silver?",
            "Looking up Silver on periodic table. Initial query used invalid column 'sym'. Correcting to 'symbol'.",
            "sql: SELECT atomic_number, sym FROM periodic_table WHERE name = 'Silver';",
            "[SQL Execution Error: no such column: sym]\n<think>\nColumn 'sym' does not exist. Using standard column name 'symbol'.\n</think>\n<act>\nsql: SELECT atomic_number, symbol FROM periodic_table WHERE name = 'Silver';\n</act>\nSilver (Ag) has an atomic number of 47."
        ),
        (
            "What is the unit and value for the Speed of Light?",
            "Querying physical constant. Initial query missed table alias. Correcting to 'physical_constants'.",
            "sql: SELECT value, unit FROM constants WHERE name = 'Speed of Light';",
            "[SQL Execution Error: no such table: constants]\n<think>\nTable 'constants' not found. Re-routing query to 'physical_constants'.\n</think>\n<act>\nsql: SELECT value_str, unit FROM physical_constants WHERE name LIKE '%Light%';\n</act>\nThe speed of light in vacuum is 299,792,458 m/s."
        ),
        (
            "What is the past tense of the verb 'speak'?",
            "Looking up irregular verb 'speak'. Initial query used incorrect column name. Correcting to 'past_simple'.",
            "sql: SELECT past FROM irregular_verbs WHERE base_form = 'speak';",
            "[SQL Execution Error: no such column: past]\n<think>\nColumn 'past' is invalid in schema. Using 'past_simple' and 'past_participle'.\n</think>\n<act>\nsql: SELECT past_simple, past_participle FROM irregular_verbs WHERE base_form = 'speak';\n</act>\nThe past tense of 'speak' is 'spoke' (past participle: 'spoken')."
        ),
    ]

    for q, th, act, rest in error_scenarios:
        for prefix in ("", "User: "):
            samples.append({
                "prompt": f"{prefix}{q}".strip(),
                "response": f"<think>\n{th}\n</think>\n<act>\n{act}\n</act>\n{rest}",
                "category": "error_recovery_traces",
                "temporal_gate": [1, 1, 1],
            })

    current_count = len(samples)
    if current_count < target_count:
        base_samples = list(samples)
        for _ in range(target_count - current_count):
            item = random.choice(base_samples).copy()
            samples.append(item)

    return samples[:target_count]


# ==============================================================================
# MAIN DATASET COMPILER
# ==============================================================================

def build_chat_thinking_dataset():
    print("\n" + "=" * 80)
    print("🧠 CEREBROSPHEROID V5: COMPILING CHAT & DATABASE-AUGMENTED THINKING DATASET")
    print("=" * 80)

    # 1. Conversational Chat & Empathy (6,000 samples)
    print("📦 Generating Conversational Chat, Empathy, and Everyday Definitions (~6,000)...")
    chat_samples = generate_conversational_chat(6000)
    print(f"   ✓ Generated {len(chat_samples):,} conversational dialogue samples.")

    # 2. Database SQL Fact Retrieval (3,000 samples)
    print("📦 Generating Database SQL Queries and Factual Retrievals (~3,000)...")
    db_samples = generate_database_fact_queries(3000)
    print(f"   ✓ Generated {len(db_samples):,} database fact query samples.")

    # 3. Calculator & Arithmetic (2,250 samples)
    print("📦 Generating Calculator & Arithmetic Reasoning (~2,250)...")
    calc_samples = generate_calculator_math(2250)
    print(f"   ✓ Generated {len(calc_samples):,} calculator math samples.")

    # 4. Python Code Synthesis (2,250 samples)
    print("📦 Generating Python Software & Algorithmic Synthesis (~2,250)...")
    code_samples = generate_python_code_samples(2250)
    print(f"   ✓ Generated {len(code_samples):,} Python code synthesis samples.")

    # 5. Agentic Modalities (1,500 samples)
    print("📦 Generating Agentic Modalities (Tools, Regex, Shell, Diagrams) (~1,500)...")
    agent_samples = generate_agentic_modalities(1500)
    print(f"   ✓ Generated {len(agent_samples):,} agentic modality samples.")

    # 6. Tool Error Recovery & Self-Correction (500 samples)
    print("📦 Generating Tool Error Recovery & Self-Correction Traces (~500)...")
    error_samples = generate_error_recovery_traces(500)
    print(f"   ✓ Generated {len(error_samples):,} error recovery self-correction samples.")

    all_samples = chat_samples + db_samples + calc_samples + code_samples + agent_samples + error_samples
    random.shuffle(all_samples)

    total_samples = len(all_samples)
    print(f"\n📊 Total Synthesized Samples: {total_samples:,}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item) + "\n")

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"💾 Written to: {OUTPUT_PATH} ({file_size_mb:.2f} MB)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    build_chat_thinking_dataset()
