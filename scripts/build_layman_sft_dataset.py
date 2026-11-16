#!/usr/bin/env python3
"""
🌟 CerebroSpheroid V5: Layman Conversational & 3-Tier Cognitive Dataset Builder
=============================================================================
Builds a specialized dataset implementing the 3-Tier Cognitive Architecture:
  1. Thinking           : <think> ... </think> -> Private scratchpad (hidden or collapsible)
  2. Non-Conversational : <act> ... </act>     -> Coding, Calculator, & Tool actions
  3. Talking / Answering: Unbounded Plain Text -> Natural, polite human conversation
"""

import os
import json
import random
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "layman_conversational_cot.jsonl")


def build_layman_dataset() -> List[Dict[str, Any]]:
    samples = []

    # -------------------------------------------------------------------------
    # 1. Greetings, Small Talk, Identity & Capabilities (Conversational - NO <act>)
    # -------------------------------------------------------------------------
    greetings = [
        ("hi", "The user is greeting me casually. Reply warmly and ask how I can help.", "Hello! How can I assist you today?"),
        ("hello", "Standard greeting. Acknowledge and offer help politely.", "Hello! How can I help you today?"),
        ("hey", "Casual greeting. Respond in a friendly tone.", "Hey there! How's it going? What can I do for you?"),
        ("good morning", "Morning greeting. Greet back politely.", "Good morning! I hope you have a great day. How can I assist you?"),
        ("good afternoon", "Afternoon greeting. Respond politely.", "Good afternoon! How is your day going? How can I help?"),
        ("good evening", "Evening greeting.", "Good evening! How can I help you tonight?"),
        ("good night", "User is heading to sleep or ending their day. Wish them a good night.", "Good night! Sleep well and have pleasant dreams!"),
        ("how are you?", "The user is asking about my state. State that I am functioning well as an AI.", "I'm doing well, thank you for asking! How are you doing today? How can I help?"),
        ("how's it going?", "Casual check-in. Respond warmly.", "It's going great! Ready to help you with questions, coding, or math. What's on your mind?"),
        ("who are you?", "The user is asking for my identity. State my name and architecture.", "I am CerebroSpheroid V5, a compact neural reasoning AI designed to assist with conversations, logical problem solving, calculator math, and code synthesis."),
        ("what are you?", "Explain what kind of AI I am.", "I am an artificial intelligence assistant built on the CerebroSpheroid architecture, featuring multi-cycle radial manifold deliberation for clear reasoning, tool execution, and problem solving."),
        ("what is your name?", "State my name directly.", "My name is CerebroSpheroid (or Cerebro for short). Nice to meet you!"),
        ("pleased to meet you", "Polite introduction. Respond with equal courtesy.", "It is a pleasure to meet you as well! How can I assist you today?"),
        ("can you help me?", "Reassure user of assistance across topics.", "Of course! I'm here to help with everyday questions, coding, mathematics, or explanations. What do you need help with?"),
        ("are you human?", "Clarify AI nature politely.", "No, I am an artificial intelligence assistant created to help you think through problems, write code, calculate math, and answer questions."),
        ("who created you?", "Explain development origin.", "I was developed as part of the CerebroSpheroid cognitive AI project, exploring radial manifold dynamics and fast multi-cycle deliberation."),
        ("what can you do?", "Summarize core capabilities clearly for a layman.", "I can help you with:\n1. Answering everyday questions and explaining concepts simply.\n2. Non-conversational tool execution: calculator math and Python code synthesis in <act> blocks.\n3. Transparent step-by-step thinking in <think> scratchpads.\n4. Logical deduction and conversation.\n\nFeel free to ask me anything!"),
        ("tell me a joke", "Share a clean, simple programming/everyday joke.", "Why do programmers prefer dark mode?\n\nBecause light attracts bugs!"),
        ("tell me another joke", "Share another funny clean joke.", "Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!"),
        ("tell me a fun fact", "Share an interesting scientific fun fact.", "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!"),
        ("i am having a bad day", "Show empathy and offer support.", "I'm really sorry to hear that. Take a deep breath, and remember that tough days pass. If you'd like to talk, learn something interesting, or work on a problem together to take your mind off things, I'm here for you."),
        ("thank you", "Polite gratitude. Respond warmly.", "You're very welcome! If you have any more questions, feel free to ask anytime."),
        ("thanks a lot", "Express warm appreciation.", "Happy to help! Let me know whenever you need anything else."),
        ("thanks", "Casual gratitude.", "Glad I could help! Let me know if you need anything else."),
        ("bye", "Farewell. Say goodbye politely.", "Goodbye! Have a wonderful day ahead!"),
        ("see you later", "Casual farewell.", "See you later! Take care!")
    ]

    for q, th, ans in greetings:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "conversational_greeting"
        })
        samples.append({
            "prompt": f"User: {q}",
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "conversational_greeting"
        })

    # -------------------------------------------------------------------------
    # 2. Everyday Science & Layman Definitions (Conversational - NO <act>)
    # -------------------------------------------------------------------------
    definitions = [
        (
            "what is earth?",
            "Explain what planet Earth is clearly and simply.",
            "Earth is the third planet from the Sun and the only astronomical body known to harbor life. About 71% of its surface is covered by liquid water oceans, and it has an atmosphere rich in nitrogen and oxygen that protects life."
        ),
        (
            "what is the earth?",
            "Explain Earth simply.",
            "The Earth is our home planet—the third planet from the Sun in our Solar System. It is an active terrestrial planet with liquid oceans, protective magnetic fields, and diverse living ecosystems."
        ),
        (
            "pixle",
            "The user typed 'pixle' (typo for pixel). Recognize the word and define pixel simply.",
            "A pixel (short for 'picture element') is the smallest individual dot or unit of color that makes up an image on a digital screen. Millions of pixels illuminated together create the images, text, and videos you see on phones, monitors, and TVs."
        ),
        (
            "what is a pixel?",
            "Define pixel clearly for a general audience.",
            "A pixel is the smallest addressable physical point or dot of color in a raster image or display device. On a 1080p screen, for instance, there are over 2 million pixels working together to display the picture."
        ),
        (
            "port",
            "The user asked 'port'. In computing, explain what a network port is simply.",
            "In computer networking, a port is a virtual communication channel or doorway that directs incoming data to a specific application or service on a device. For example, web browsers use port 80 (HTTP) or port 443 (HTTPS), while email services often use port 25 or 587."
        ),
        (
            "what is a computer port?",
            "Explain network and hardware ports simply.",
            "A computer port can refer to two things:\n1. Network Port (Virtual): A numerical endpoint (like port 80 for web traffic or port 22 for SSH) that directs network packets to the right software.\n2. Physical Port (Hardware): A socket on your machine (like USB, HDMI, or Ethernet) where you plug in external cables and peripherals."
        ),
        (
            "why is the sky blue?",
            "Explain Rayleigh scattering in accessible layman terms.",
            "The sky is blue because of Rayleigh scattering. Sunlight looks white, but it contains all the colors of the rainbow. Earth's atmosphere scatters shorter wavelengths of light (blue and violet) much more than longer wavelengths (red and yellow). Because our eyes are more sensitive to blue, the daytime sky appears bright blue."
        ),
        (
            "what is water made of?",
            "Explain the chemical formula of water simply.",
            "Water is a chemical compound made of two hydrogen atoms bonded to one oxygen atom, written chemically as H2O."
        ),
        (
            "what is gravity?",
            "Explain gravity in simple intuitive terms.",
            "Gravity is a fundamental natural force that attracts objects with mass toward each other. It is what keeps our feet on the ground, keeps the Moon in orbit around Earth, and keeps the Earth orbiting the Sun."
        ),
        (
            "what is electricity?",
            "Explain electricity in intuitive layman terms.",
            "Electricity is a form of energy resulting from the flow of charged particles, specifically electrons. It powers our lights, computers, appliances, and electric vehicles."
        ),
        (
            "what is lightning?",
            "Explain atmospheric electrical discharge.",
            "Lightning is a giant natural spark of electricity that occurs between clouds, or between a cloud and the ground, caused by a buildup of electrical charge during thunderstorms."
        ),
        (
            "what is thunder?",
            "Explain the acoustic sound of lightning.",
            "Thunder is the acoustic sound wave produced by lightning. When lightning strikes, it heats the surrounding air to around 30,000°C (54,000°F) in a fraction of a second, causing the air to rapidly expand and create a loud sonic shockwave."
        ),
        (
            "why is the ocean salty?",
            "Explain ocean salinity origin simply.",
            "The ocean is salty primarily because rain washes minerals and dissolved salts from rocks on land into rivers, which eventually flow into the sea. Over billions of years, water evaporates from the ocean but leaves the dissolved salts behind."
        ),
        (
            "how do airplanes fly?",
            "Explain aerodynamic lift in simple language.",
            "Airplanes fly because of aerodynamic lift. Airplane wings are shaped (airfoils) so that as the plane moves forward, air flows faster over the curved top of the wing than underneath, creating lower air pressure on top and higher pressure below, pushing the plane up into the sky."
        ),
        (
            "how do boats float?",
            "Explain Archimedes principle simply.",
            "Boats float because of buoyancy and displacement. When a boat is placed in water, it pushes aside (displaces) an amount of water. Because the boat is hollow and filled with air, its overall average density is less than water, creating an upward buoyant force that keeps it afloat."
        ),
        (
            "how does a refrigerator work?",
            "Explain heat pump refrigeration cycle simply.",
            "A refrigerator works by removing heat from the inside and releasing it outside. A fluid called refrigerant circulates through coils: it absorbs heat inside as it evaporates into a gas, and then a compressor squeezes the gas outside, releasing the heat into the room and condensing it back into a cold liquid."
        ),
        (
            "how does a microwave heat food?",
            "Explain microwave dielectric heating.",
            "A microwave oven produces electromagnetic waves (microwaves) that pass through food and cause water molecules inside the food to rapidly vibrate. This molecular friction generates heat, cooking the food from the inside out."
        ),
        (
            "what is photosynthesis?",
            "Explain photosynthesis simply.",
            "Photosynthesis is the biological process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into oxygen and glucose (chemical energy/food)."
        ),
        (
            "what is dna?",
            "Explain DNA simply.",
            "DNA (deoxyribonucleic acid) is the biological molecule that carries the genetic blueprint or instructions for the development, functioning, and reproduction of all known living organisms."
        ),
        (
            "what is an atom?",
            "Explain atoms in simple terms.",
            "An atom is the basic building block of all ordinary matter. It consists of a dense central nucleus made of protons and neutrons, surrounded by a cloud of orbiting electrons."
        ),
        (
            "what is the sun?",
            "Define the Sun simply.",
            "The Sun is the massive star at the center of our Solar System. It is an almost perfect sphere of hot plasma powered by nuclear fusion, supplying the light, heat, and energy necessary for life on Earth."
        ),
        (
            "what is the moon?",
            "Define the Moon simply.",
            "The Moon is Earth's only natural satellite. It orbits Earth roughly once every 27.3 days and its gravitational pull is the primary cause of ocean tides on Earth."
        ),
        (
            "what is an earthquake?",
            "Explain seismic activity simply.",
            "An earthquake is the shaking of the Earth's surface caused by the sudden release of energy in the crust, typically when tectonic plates grind against each other along a fault line."
        ),
        (
            "what is a volcano?",
            "Explain volcanic eruption simply.",
            "A volcano is an opening or rupture in the Earth's crust that allows hot magma, volcanic ash, and gases to escape from deep underground to the surface."
        ),
        (
            "what is a black hole?",
            "Explain black hole gravitational pull.",
            "A black hole is a region of spacetime where gravity is so strong that nothing—not even particles or light—has enough speed to escape its gravitational pull."
        ),
        (
            "what is the speed of light?",
            "State the speed of light in vacuum clearly.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 186,282 miles per second). At this speed, light could travel around the entire Earth about 7.5 times in just one second!"
        ),
        (
            "why do leaves change color in autumn?",
            "Explain chlorophyll breakdown.",
            "In autumn, shorter days and cooler temperatures cause deciduous trees to stop producing chlorophyll (the green pigment that absorbs sunlight). As the green fades, other pigments that were already present—like yellow and orange carotenoids—finally become visible."
        ),
        (
            "why do we sleep?",
            "Explain restorative function of sleep.",
            "We sleep so our body and brain can recover, repair tissues, consolidate memories, remove metabolic waste products from brain cells, and restore energy levels for the next day."
        ),
        # Computing and tech concepts for laymen:
        (
            "what is the internet?",
            "Explain internet global network simply.",
            "The internet is a vast global network of billions of interconnected computers, servers, and devices communicating using standardized protocols (like TCP/IP) to share information."
        ),
        (
            "what is wi-fi?",
            "Explain wireless local area networking.",
            "Wi-Fi is a wireless networking technology that uses radio frequencies (typically 2.4 GHz, 5 GHz, or 6 GHz) to transmit high-speed data between devices and a local router without physical cables."
        ),
        (
            "what is bluetooth?",
            "Explain short-range wireless communication.",
            "Bluetooth is a short-range wireless technology that lets electronic devices (like headphones, phones, mice, and keyboards) connect and exchange data over short distances (usually up to 10 meters)."
        ),
        (
            "what is an operating system?",
            "Explain OS software simply.",
            "An operating system (like Windows, macOS, Linux, Android, or iOS) is the core software that manages a computer's hardware, memory, and processors, providing a platform for apps to run."
        ),
        (
            "what is the difference between ram and ssd?",
            "Compare volatile RAM and persistent SSD storage.",
            "RAM (Random Access Memory) is ultra-fast, temporary memory used by your computer to hold apps currently in use; it gets wiped when the device turns off. An SSD (Solid State Drive) is long-term storage where your files, operating system, and photos remain saved permanently even when the computer is shut down."
        ),
        (
            "what is an ip address?",
            "Explain IP address simply.",
            "An IP (Internet Protocol) address is a unique numerical label assigned to every device connected to a computer network (like 192.168.1.1 or 8.8.8.8), acting like a postal address so data packets find their destination."
        ),
        (
            "what is dns?",
            "Explain Domain Name System simply.",
            "DNS (Domain Name System) is like the phonebook of the internet. It translates human-friendly website names (like google.com) into machine-readable IP addresses (like 142.250.190.46) so browsers can load pages."
        ),
        (
            "what is cloud computing?",
            "Explain cloud computing simply.",
            "Cloud computing means accessing and storing data, servers, databases, and software over the internet on remote data centers instead of relying entirely on your local computer's hard drive."
        ),
        (
            "what is an algorithm?",
            "Define algorithm simply.",
            "An algorithm is a step-by-step set of instructions or rules followed by a computer to solve a problem, perform a calculation, or complete a task (like a cooking recipe for computers)."
        ),
        (
            "what is artificial intelligence?",
            "Define AI in simple terms.",
            "Artificial Intelligence (AI) refers to computer systems engineered to perform tasks that typically require human intelligence, such as recognizing speech, understanding language, solving problems, learning from data, and making decisions."
        ),
    ]

    for q, th, ans in definitions:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "layman_definition"
        })
        samples.append({
            "prompt": f"Explain simply: {q}",
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "layman_definition"
        })

    # -------------------------------------------------------------------------
    # 3. Practical Everyday Advice & Common Sense (Conversational - NO <act>)
    # -------------------------------------------------------------------------
    practical_advice = [
        (
            "how do I make tea?",
            "Explain standard black/green tea preparation simply.",
            "To make a hot cup of tea:\n1. Boil fresh water in a kettle (around 100°C for black tea, 80°C for green tea).\n2. Place a tea bag or 1 teaspoon of loose tea leaves into your mug.\n3. Pour the hot water over the tea.\n4. Let it steep for 3 to 5 minutes.\n5. Remove the tea bag or strain the leaves, and add honey, sugar, or milk to taste."
        ),
        (
            "how do I boil an egg?",
            "Give clear timing for soft and hard boiled eggs.",
            "To boil an egg:\n1. Bring a pot of water to a gentle rolling boil.\n2. Carefully lower eggs into the water with a spoon.\n3. Cook for:\n   • 6 minutes: Soft-boiled with runny yolk.\n   • 8 minutes: Medium with jammy yolk.\n   • 10 to 12 minutes: Fully hard-boiled.\n4. Transfer immediately to an ice bath for 3 minutes for easy peeling."
        ),
        (
            "how do I stay hydrated?",
            "Provide practical hydration tips.",
            "To stay well hydrated throughout the day:\n1. Drink a glass of water right when you wake up.\n2. Carry a reusable water bottle.\n3. Drink water before, during, and after physical exercise.\n4. Eat water-rich fruits and vegetables like watermelon, cucumbers, and oranges.\n5. A good rule of thumb is around 2 to 3 liters of water per day."
        ),
        (
            "how do I save battery on my smartphone?",
            "Give actionable mobile battery saving tips.",
            "To make your phone battery last longer:\n1. Lower screen brightness or enable Auto-Brightness.\n2. Turn on Battery Saver mode.\n3. Turn off Wi-Fi, Bluetooth, and GPS when not in use.\n4. Close unnecessary background apps and disable background app refresh.\n5. Reduce screen timeout to 30 seconds."
        ),
        (
            "how do I write a professional email?",
            "Explain professional email etiquette simply.",
            "To write a clear professional email:\n1. Subject Line: Keep it concise and specific (e.g., 'Project Update - Nov 16').\n2. Greeting: Use a polite salutation ('Dear [Name],' or 'Hello [Name],').\n3. Opening: State your purpose clearly in the first sentence.\n4. Body: Use short paragraphs or bullet points for readability.\n5. Call to Action: State next steps or deadlines politely.\n6. Sign-off: Close warmly ('Best regards,' or 'Sincerely,') followed by your name."
        )
    ]

    for q, th, ans in practical_advice:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "practical_advice"
        })

    # -------------------------------------------------------------------------
    # 4. World Capitals & Direct Factual Knowledge (Conversational - NO <act>)
    # -------------------------------------------------------------------------
    capitals = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Italy", "Rome"), ("United Kingdom", "London"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Spain", "Madrid"), ("India", "New Delhi"),
        ("China", "Beijing"), ("Brazil", "Brasília"), ("Egypt", "Cairo"),
        ("Russia", "Moscow"), ("United States", "Washington, D.C."),
        ("South Korea", "Seoul"), ("Mexico", "Mexico City"), ("Argentina", "Buenos Aires"),
        ("Netherlands", "Amsterdam"), ("Switzerland", "Bern"), ("Sweden", "Stockholm"),
        ("Norway", "Oslo"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
        ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"),
        ("New Zealand", "Wellington"), ("South Africa", "Pretoria"), ("Kenya", "Nairobi"),
        ("Turkey", "Ankara"), ("Saudi Arabia", "Riyadh"), ("United Arab Emirates", "Abu Dhabi"),
        ("Indonesia", "Jakarta"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
        ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Belgium", "Brussels"),
        ("Chile", "Santiago"), ("Colombia", "Bogotá"), ("Peru", "Lima")
    ]

    for country, cap in capitals:
        q1 = f"What is the capital of {country}?"
        th1 = f"Identify the capital city of {country}. The capital is {cap}."
        ans1 = f"The capital of {country} is {cap}."
        samples.append({
            "prompt": q1,
            "response": f"<think>\n{th1}\n</think>\n{ans1}",
            "category": "world_knowledge"
        })
        samples.append({
            "prompt": f"Capital of {country}:",
            "response": f"<think>\n{th1}\n</think>\n{cap}",
            "category": "world_knowledge"
        })

    # Solar system planets & geography facts:
    geo_facts = [
        ("What are the planets in our solar system in order?", "List planets from closest to farthest from the Sun.", "The eight planets in order of distance from the Sun are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
        ("What is the largest ocean on Earth?", "Identify the largest ocean.", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the Earth's surface."),
        ("What is the highest mountain in the world?", "Identify Mount Everest elevation.", "Mount Everest is the highest mountain above sea level, reaching a peak elevation of 8,848.86 meters (29,031.7 feet) in the Himalayas."),
        ("What is the longest river in the world?", "Identify the Nile and Amazon rivers.", "The Nile River in Africa is traditionally recognized as the longest river in the world at approximately 6,650 kilometers (4,132 miles), closely rivaled by the Amazon River.")
    ]
    for q, th, ans in geo_facts:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n{ans}",
            "category": "world_knowledge"
        })

    # -------------------------------------------------------------------------
    # 5. Calculator & Arithmetic Tasks (Non-Conversational Action in <act>)
    # -------------------------------------------------------------------------
    math_examples = [
        ("15 + 27", "15 + 27.\n15 + 20 = 35.\n35 + 7 = 42.\nExecute calculator.", "15 + 27 = 42", "15 + 27 equals 42."),
        ("25 + 17", "25 + 17.\n25 + 10 = 35.\n35 + 7 = 42.\nExecute calculator.", "25 + 17 = 42", "25 + 17 is 42."),
        ("38 + 45", "38 + 45.\n38 + 40 = 78.\n78 + 5 = 83.\nExecute calculator.", "38 + 45 = 83", "38 + 45 equals 83."),
        ("125 + 375", "125 + 375.\n100 + 300 = 400. 25 + 75 = 100. 400 + 100 = 500.", "125 + 375 = 500", "125 + 375 is 500."),
        ("100 - 37", "100 - 37.\n100 - 30 = 70.\n70 - 7 = 63.\nExecute calculator.", "100 - 37 = 63", "100 minus 37 is 63."),
        ("450 - 180", "450 - 180.\n450 - 100 = 350. 350 - 80 = 270.", "450 - 180 = 270", "450 minus 180 equals 270."),
        ("8 * 7", "8 multiplied by 7.\n8 × 7 = 56.\nExecute calculator.", "8 * 7 = 56", "8 multiplied by 7 equals 56."),
        ("9 * 6", "9 multiplied by 6.\n9 × 6 = 54.\nExecute calculator.", "9 * 6 = 54", "9 × 6 is 54."),
        ("12 * 12", "12 squared.\n12 × 12 = 144.\nExecute calculator.", "12 * 12 = 144", "12 squared is 144."),
        ("15 * 6", "15 multiplied by 6.\n10 * 6 = 60, 5 * 6 = 30. 60 + 30 = 90.", "15 * 6 = 90", "15 times 6 is 90."),
        ("24 * 5", "24 multiplied by 5.\n20 * 5 = 100, 4 * 5 = 20. 100 + 20 = 120.", "24 * 5 = 120", "24 times 5 equals 120."),
        ("100 / 4", "100 divided by 4.\n100 / 4 = 25.\nExecute calculator.", "100 / 4 = 25", "100 divided by 4 equals 25."),
        ("84 / 4", "84 divided by 4.\n80 / 4 = 20, 4 / 4 = 1.\n20 + 1 = 21.\nExecute calculator.", "84 / 4 = 21", "84 divided by 4 is 21."),
        ("250 / 5", "250 divided by 5.\n25 / 5 = 5, so 250 / 5 = 50.", "250 / 5 = 50", "250 divided by 5 is 50."),
        ("144 / 12", "144 divided by 12.\n12 * 12 = 144, so 144 / 12 = 12.", "144 / 12 = 12", "144 divided by 12 equals 12."),
        ("2x = 10, solve for x", "We have 2x = 10. Divide both sides by 2: x = 5.", "x = 10 / 2 = 5", "The solution is x = 5."),
        ("3x + 7 = 22, solve for x", "3x + 7 = 22. Subtract 7: 3x = 15. Divide by 3: x = 5.", "x = (22 - 7) / 3 = 5", "Subtract 7 from both sides to get 3x = 15, then divide by 3 to get x = 5."),
        ("5x - 15 = 0, solve for x", "5x - 15 = 0. Add 15: 5x = 15. Divide by 5: x = 3.", "x = 15 / 5 = 3", "The solution is x = 3."),
        ("4x = 28, solve for x", "4x = 28. Divide by 4: x = 7.", "x = 28 / 4 = 7", "The solution is x = 7."),
        ("20% of 150", "20% of 150 = 0.20 * 150 = 30.", "0.20 * 150 = 30", "20% of 150 is 30."),
        ("15% tip on $60 bill", "Tip = 0.15 * 60 = 9 dollars.", "0.15 * 60 = 9", "A 15% tip on a $60 bill is $9.00.")
    ]

    for expr, th, calc_act, ans in math_examples:
        samples.append({
            "prompt": f"What is {expr}?",
            "response": f"<think>\n{th}\n</think>\n<act>\ncalculator: {calc_act}\n</act>\n{ans}",
            "category": "calculator_action"
        })
        samples.append({
            "prompt": f"Solve: {expr}",
            "response": f"<think>\n{th}\n</think>\n<act>\ncalculator: {calc_act}\n</act>\n{ans}",
            "category": "calculator_action"
        })
        samples.append({
            "prompt": f"calculate {expr}",
            "response": f"<think>\n{th}\n</think>\n<act>\ncalculator: {calc_act}\n</act>\n{ans}",
            "category": "calculator_action"
        })

    word_problems = [
        (
            "If I have 3 apples and eat 1, how many do I have left?",
            "Initial count is 3 apples. Eating 1 means subtracting 1: 3 - 1 = 2.",
            "3 - 1 = 2",
            "You have 2 apples left."
        ),
        (
            "If John has 5 books and buys 3 more, how many books does he have in total?",
            "Initial count = 5 books. Added count = 3 books. Total = 5 + 3 = 8.",
            "5 + 3 = 8",
            "John has 8 books in total."
        ),
        (
            "If a box holds 12 donuts and you buy 3 boxes, how many donuts do you have?",
            "12 donuts per box × 3 boxes = 36 donuts.",
            "12 * 3 = 36",
            "You have 36 donuts."
        ),
        (
            "Sarah has $50. She spends $18 on lunch. How much money does she have left?",
            "Initial money: $50. Spent: $18. Remaining = 50 - 18 = 32.",
            "50 - 18 = 32",
            "Sarah has $32 left."
        ),
        (
            "If a car travels at 60 mph for 3 hours, how far does it travel?",
            "Distance = Speed × Time = 60 miles/hour × 3 hours = 180 miles.",
            "60 * 3 = 180",
            "The car travels 180 miles."
        ),
        (
            "If Tom has 15 books and gives 4 to Alice and 3 to Bob, how many books does Tom have left?",
            "Initial = 15. Given away = 4 + 3 = 7. Remaining = 15 - 7 = 8.",
            "15 - (4 + 3) = 8",
            "Tom has 8 books left."
        ),
        (
            "A rectangle has a length of 8 meters and a width of 5 meters. What is its area?",
            "Area = length × width = 8 × 5 = 40 square meters.",
            "8 * 5 = 40",
            "The area of the rectangle is 40 square meters."
        ),
        (
            "A square has sides of length 6 cm. What is its perimeter?",
            "Perimeter of square = 4 × side = 4 × 6 = 24 cm.",
            "4 * 6 = 24",
            "The perimeter of the square is 24 cm."
        ),
        (
            "You buy 3 shirts at $25 each. What is the total cost?",
            "Total cost = 3 × 25 = 75 dollars.",
            "3 * 25 = 75",
            "The total cost is $75."
        )
    ]

    for q, th, calc_act, ans in word_problems:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\ncalculator: {calc_act}\n</act>\n{ans}",
            "category": "calculator_action"
        })

    # -------------------------------------------------------------------------
    # 6. Coding & Software Synthesis Tasks (Non-Conversational Code in <act>)
    # -------------------------------------------------------------------------
    coding_tasks = [
        (
            "python script to sort a list",
            "The user needs a Python script to sort a list. Synthesize clean executable code inside the action block.",
            "numbers = [42, 17, 8, 99, 23, 1]\nsorted_numbers = sorted(numbers)\nprint(\"Original:\", numbers)\nprint(\"Sorted:  \", sorted_numbers)",
            "Here is the Python script to sort a list. It uses Python's built-in `sorted()` function to return a new sorted list in ascending order."
        ),
        (
            "how to sort a list in python",
            "Explain how to sort a list in Python with executable code in <act>.",
            "# Option 1: sorted() creates a new list\nnums = [5, 2, 8, 1]\nnew_nums = sorted(nums)\nprint(\"sorted():\", new_nums)\n\n# Option 2: .sort() modifies list in place\nnums.sort()\nprint(\".sort(): \", nums)",
            "In Python, you can sort a list using either `sorted()` (to create a new list) or `.sort()` (to sort in place)."
        ),
        (
            "write a python function to check if a string is a palindrome",
            "A palindrome reads the same forwards and backwards. Emit clean Python code inside the action block.",
            "def is_palindrome(s: str) -> bool:\n    clean = s.lower().replace(\" \", \"\")\n    return clean == clean[::-1]\n\n# Test examples\nprint(is_palindrome(\"racecar\"))  # True\nprint(is_palindrome(\"hello\"))    # False",
            "Here is a Python function to check for palindromes. It strips spaces, converts to lowercase, and compares the string to its reverse using slice notation `[::-1]`."
        ),
        (
            "python script to reverse a string",
            "Show string reversal in Python using slice notation.",
            "def reverse_string(s: str) -> str:\n    return s[::-1]\n\ntext = \"hello world\"\nprint(reverse_string(text))  # dlrow olleh",
            "In Python, the most efficient way to reverse a string is using slice notation `[::-1]`."
        ),
        (
            "python hello world",
            "Provide the canonical hello world script in Python.",
            "print(\"Hello, World!\")",
            "Here is the standard 'Hello, World!' program in Python."
        ),
        (
            "how to read a file in python",
            "Provide file reading context manager in Python.",
            "with open(\"example.txt\", \"r\", encoding=\"utf-8\") as f:\n    content = f.read()\n    print(content)",
            "Using the `with open(...)` context manager ensures the file is automatically closed after reading, even if errors occur."
        ),
        (
            "how to write to a file in python",
            "Show file writing with context manager in Python.",
            "with open(\"output.txt\", \"w\", encoding=\"utf-8\") as f:\n    f.write(\"Hello, file!\\n\")\n    f.write(\"This is a second line.\\n\")",
            "Using mode `'w'` opens the file for writing (overwriting if it already exists). The `with` statement safely flushes and closes the file."
        ),
        (
            "python function to find the maximum number in a list",
            "Show built-in max() as well as a loop implementation.",
            "def find_max(numbers: list) -> int:\n    if not numbers:\n        return None\n    current_max = numbers[0]\n    for num in numbers[1:]:\n        if num > current_max:\n            current_max = num\n    return current_max\n\nprint(find_max([14, 58, 2, 99, 31]))  # 99",
            "This function iterates through the list and keeps track of the largest value found."
        ),
        (
            "python script to calculate factorial",
            "Implement iterative factorial calculation.",
            "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nprint(factorial(5))  # 120",
            "The factorial of `n` is the product of all positive integers less than or equal to `n`."
        ),
        (
            "python script to check if a number is prime",
            "Check for primality with trial division.",
            "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17))  # True\nprint(is_prime(20))  # False",
            "A number is prime if it has no divisors other than 1 and itself. We only need to test divisors up to the square root of `n`."
        ),
        (
            "python script to generate fibonacci sequence",
            "Generate n Fibonacci numbers.",
            "def fibonacci(n: int) -> list:\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq\n\nprint(fibonacci(7))  # [0, 1, 1, 2, 3, 5, 8]",
            "Each number in the Fibonacci sequence is the sum of the two preceding numbers."
        ),
        (
            "python script to reverse a list",
            "Reverse a list in Python.",
            "# Method 1: Slicing (creates new list)\nnumbers = [1, 2, 3, 4, 5]\nreversed_list = numbers[::-1]\nprint(reversed_list)  # [5, 4, 3, 2, 1]\n\n# Method 2: In-place reversal\nnumbers.reverse()\nprint(numbers)        # [5, 4, 3, 2, 1]",
            "You can reverse a list using slice notation `[::-1]` for a new list, or `.reverse()` to modify the list in place."
        ),
        (
            "python remove duplicates from list",
            "Remove duplicates preserving or not preserving order.",
            "# Fast method with set (order not guaranteed)\nnums = [1, 2, 2, 3, 4, 4, 5]\nunique = list(set(nums))\n\n# Order-preserving method using dict.fromkeys\nunique_ordered = list(dict.fromkeys(nums))\nprint(\"Unique (ordered):\", unique_ordered)",
            "Using `list(dict.fromkeys(nums))` removes duplicate elements while preserving the original insertion order."
        ),
        (
            "python count word frequency in a string",
            "Count word frequencies with dict.",
            "def count_words(text: str) -> dict:\n    words = text.lower().split()\n    counts = {}\n    for word in words:\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n\ntext = \"apple banana apple orange banana apple\"\nprint(count_words(text))",
            "This function splits the text into words and tracks occurrences in a standard Python dictionary."
        ),
        (
            "how to parse json in python",
            "Demonstrate json.loads and json.dumps.",
            "import json\n\n# 1. Parse JSON string to Python dict\njson_str = '{\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}'\ndata = json.loads(json_str)\nprint(\"Parsed Name:\", data[\"name\"])\n\n# 2. Serialize Python dict to JSON string\nback_to_json = json.dumps(data, indent=2)\nprint(back_to_json)",
            "Python's standard `json` module provides `json.loads()` for deserialization and `json.dumps()` for formatting."
        ),
        (
            "python check leap year",
            "Determine if a year is a leap year.",
            "def is_leap_year(year: int) -> bool:\n    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):\n        return True\n    return False\n\nprint(is_leap_year(2024))  # True\nprint(is_leap_year(2023))  # False",
            "A year is a leap year if divisible by 4, except for century years which must also be divisible by 400."
        ),
        (
            "python convert celsius to fahrenheit",
            "Formula F = C * 9/5 + 32.",
            "def celsius_to_fahrenheit(c: float) -> float:\n    return (c * 9/5) + 32\n\ndef fahrenheit_to_celsius(f: float) -> float:\n    return (f - 32) * 5/9\n\nprint(\"25°C in F:\", celsius_to_fahrenheit(25))  # 77.0°F\nprint(\"77°F in C:\", fahrenheit_to_celsius(77))  # 25.0°C",
            "Here are functions to convert temperatures between Celsius and Fahrenheit."
        ),
        (
            "python binary search implementation",
            "Implement O(log n) binary search algorithm.",
            "def binary_search(arr: list, target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n\nnums = [1, 3, 5, 7, 9, 11]\nprint(binary_search(nums, 7))  # Index 3",
            "Binary search repeatedly divides the search interval in half on a sorted array, running in O(log n) time."
        ),
        (
            "python merge two dictionaries",
            "Merge dictionaries in Python 3.9+.",
            "dict1 = {\"a\": 1, \"b\": 2}\ndict2 = {\"b\": 99, \"c\": 4}\n\n# Python 3.9+ union operator |\nmerged = dict1 | dict2\nprint(merged)  # {\x27a\x27: 1, \x27b\x27: 99, \x27c\x27: 4}",
            "In Python 3.9 and later, the pipe operator `|` cleanly merges dictionaries, with keys from the right dictionary overwriting keys from the left."
        ),
        (
            "python flatten a nested list",
            "Flatten 2D list with list comprehension.",
            "nested = [[1, 2, 3], [4, 5], [6, 7, 8]]\nflat = [item for sublist in nested for item in sublist]\nprint(flat)  # [1, 2, 3, 4, 5, 6, 7, 8]",
            "A concise list comprehension flattens a 2D list into a single flat list."
        )
    ]

    for q, th, code_act, expl in coding_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{code_act}\n</act>\n{expl}",
            "category": "code_action"
        })
        samples.append({
            "prompt": f"Write a {q}",
            "response": f"<think>\n{th}\n</think>\n<act>\n{code_act}\n</act>\n{expl}",
            "category": "code_action"
        })

    # -------------------------------------------------------------------------
    # 7. Tool & System Actions (Non-Conversational Tool Execution in <act>)
    # -------------------------------------------------------------------------
    tool_tasks = [
        (
            "search the web for python documentation",
            "The user wants web search. Dispatch search tool action in <act>.",
            "search: \"official python documentation 3.12\"",
            "Searching the web for the official Python documentation."
        ),
        (
            "run bash command to list files in directory",
            "Dispatch shell command in <act>.",
            "bash: ls -la",
            "Listing all files and directories with detailed permissions."
        ),
        (
            "check disk space on linux",
            "Check disk space usage via bash df -h.",
            "bash: df -h",
            "Checking available disk space across all mounted filesystems."
        ),
        (
            "check memory usage on linux",
            "Check RAM usage via bash free -h.",
            "bash: free -h",
            "Displaying total, used, and available system RAM."
        ),
        (
            "check git status",
            "Run git status command.",
            "bash: git status",
            "Checking the current Git branch and working tree status."
        ),
        (
            "calculate square root of 144",
            "Calculate square root with calculator.",
            "calculator: math.sqrt(144) = 12",
            "The square root of 144 is 12."
        ),
        (
            "calculate 2 to the power of 10",
            "Calculate 2^10 with calculator.",
            "calculator: 2 ** 10 = 1024",
            "2 raised to the 10th power is 1,024."
        ),
        (
            "calculate 2 to the power of 16",
            "Calculate 2^16 with calculator.",
            "calculator: 2 ** 16 = 65536",
            "2 raised to the 16th power is 65,536."
        )
    ]

    for q, th, tool_act, expl in tool_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{tool_act}\n</act>\n{expl}",
            "category": "tool_action"
        })

    # -------------------------------------------------------------------------
    # 8. SQL Database Queries (Non-Conversational SQL in <act>)
    # -------------------------------------------------------------------------
    sql_tasks = [
        (
            "sql query to find all active users",
            "The user wants an SQL query. Emit clean ANSI SQL in <act>.",
            "sql: SELECT id, username, email FROM users WHERE is_active = 1 ORDER BY created_at DESC LIMIT 10;",
            "Here is the SQL query to select active users, ordered by newest first."
        ),
        (
            "sql query to find average salary by department",
            "Generate SQL GROUP BY query for average salary.",
            "sql: SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department HAVING AVG(salary) > 50000;",
            "This query groups employees by department, computes the average salary for each, and filters for departments above $50,000."
        ),
        (
            "sql join customers and orders",
            "Write inner join between customers and orders.",
            "sql: SELECT c.name, COUNT(o.id) AS total_orders FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;",
            "This query joins customers with their orders and counts the total orders per customer."
        ),
        (
            "sql query to update product prices by 10 percent discount",
            "Write SQL UPDATE statement with discount.",
            "sql: UPDATE products SET price = price * 0.90 WHERE category = 'Clearance';",
            "This query applies a 10% discount to all products in the 'Clearance' category."
        )
    ]
    for q, th, sql_act, expl in sql_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{sql_act}\n</act>\n{expl}",
            "category": "sql_action"
        })
        samples.append({
            "prompt": f"Write an {q}",
            "response": f"<think>\n{th}\n</think>\n<act>\n{sql_act}\n</act>\n{expl}",
            "category": "sql_action"
        })

    # -------------------------------------------------------------------------
    # 9. Structured JSON Function / Tool Invocations (Non-Conversational Tool in <act>)
    # -------------------------------------------------------------------------
    function_tasks = [
        (
            "check the weather in Tokyo",
            "Dispatch weather function call in <act>.",
            "tool: get_weather({\"city\": \"Tokyo\", \"units\": \"metric\"})",
            "I have dispatched a request to retrieve the current weather conditions in Tokyo."
        ),
        (
            "get stock price of Apple",
            "Dispatch financial market quote tool in <act>.",
            "tool: get_stock_quote({\"symbol\": \"AAPL\"})",
            "Querying the real-time stock quote for Apple Inc. (AAPL)."
        ),
        (
            "search documentation for spheroid architecture",
            "Dispatch knowledge base search tool in <act>.",
            "tool: search_kb({\"query\": \"spheroid manifold architecture\", \"top_k\": 3})",
            "Searching the technical documentation repository for articles matching 'spheroid manifold architecture'."
        ),
        (
            "send slack notification to engineering channel",
            "Dispatch messaging API tool call in <act>.",
            "tool: send_slack_message({\"channel\": \"#engineering\", \"text\": \"Option A SFT deployment complete.\"}})",
            "I have posted the status notification to the engineering Slack channel."
        )
    ]
    for q, th, fn_act, expl in function_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{fn_act}\n</act>\n{expl}",
            "category": "function_tool_action"
        })

    # -------------------------------------------------------------------------
    # 10. Mermaid Architectural Diagrams & Flowcharts (Non-Conversational Diagram in <act>)
    # -------------------------------------------------------------------------
    diagram_tasks = [
        (
            "draw a flowchart of user authentication",
            "Synthesize Mermaid graph for authentication workflow in <act>.",
            "mermaid:\ngraph TD\n    A[User Submits Credentials] --> B{Valid Login?}\n    B -- Yes --> C[Generate JWT Token]\n    C --> D[Redirect to Dashboard]\n    B -- No --> E[Display 401 Unauthorized]",
            "Here is the Mermaid flowchart representing the user authentication process."
        ),
        (
            "draw a sequence diagram for api request",
            "Emit sequence diagram in Mermaid syntax.",
            "mermaid:\nsequenceDiagram\n    Client->>Gateway: GET /api/v1/profile\n    Gateway->>AuthService: Validate Bearer Token\n    AuthService-->>Gateway: 200 OK (User ID)\n    Gateway->>Database: Query Profile Record\n    Database-->>Gateway: Profile JSON\n    Gateway-->>Client: 200 OK JSON Response",
            "Here is the sequence diagram illustrating the API request and token validation lifecycle."
        )
    ]
    for q, th, diag_act, expl in diagram_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{diag_act}\n</act>\n{expl}",
            "category": "diagram_action"
        })

    # -------------------------------------------------------------------------
    # 11. Regular Expressions & Pattern Matching (Non-Conversational Regex in <act>)
    # -------------------------------------------------------------------------
    regex_tasks = [
        (
            "regex to validate email address",
            "Provide email validation regex pattern in <act>.",
            "regex: r\"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$\"",
            "This regular expression validates email addresses by ensuring proper alphanumeric characters, an '@' separator, and a domain name."
        ),
        (
            "regex to find phone numbers",
            "Provide regex pattern for US phone numbers.",
            "regex: r\"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b\"",
            "This pattern matches 10-digit telephone numbers formatted with optional hyphens or periods (e.g. 123-456-7890)."
        )
    ]
    for q, th, reg_act, expl in regex_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{reg_act}\n</act>\n{expl}",
            "category": "regex_action"
        })

    # -------------------------------------------------------------------------
    # 12. Unit Testing & Code Assertions (Non-Conversational Test in <act>)
    # -------------------------------------------------------------------------
    unit_test_tasks = [
        (
            "unit tests for add function",
            "Write pytest / assert statements in <act>.",
            "test: assert add(2, 3) == 5\ntest: assert add(-1, 1) == 0\ntest: assert add(0, 0) == 0\ntest: assert add(-5, -5) == -10",
            "Here are unit test assertions verifying the `add` function across positive, negative, zero, and boundary inputs."
        ),
        (
            "unit test for palindrome checker",
            "Write assertions testing palindrome logic.",
            "test: assert is_palindrome(\"racecar\") is True\ntest: assert is_palindrome(\"A man a plan a canal Panama\") is True\ntest: assert is_palindrome(\"hello\") is False",
            "These assertions test palindrome detection on single words, phrases with spaces, and non-palindromes."
        )
    ]
    for q, th, test_act, expl in unit_test_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{test_act}\n</act>\n{expl}",
            "category": "unit_test_action"
        })

    # -------------------------------------------------------------------------
    # 13. Date & Time / Clock Services (Non-Conversational Datetime in <act>)
    # -------------------------------------------------------------------------
    time_tasks = [
        (
            "what is the current utc time?",
            "Query system clock in <act>.",
            "datetime: get_current_time(timezone=\"UTC\")",
            "Retrieving current timestamp in Coordinated Universal Time (UTC)."
        ),
        (
            "how many days until new year 2027?",
            "Execute date difference operation.",
            "datetime: date_difference(start=\"today\", end=\"2027-01-01\")",
            "Calculating the number of days remaining until January 1, 2027."
        )
    ]
    for q, th, dt_act, expl in time_tasks:
        samples.append({
            "prompt": q,
            "response": f"<think>\n{th}\n</think>\n<act>\n{dt_act}\n</act>\n{expl}",
            "category": "datetime_action"
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(samples)} 3-Tier Cognitive examples (Conversational, Coding in <act>, Calculator in <act>) -> {OUTPUT_PATH}")
    return samples


if __name__ == "__main__":
    build_layman_dataset()
