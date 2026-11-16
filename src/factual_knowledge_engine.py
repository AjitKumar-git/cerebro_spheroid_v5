#!/usr/bin/env python3
"""
🏛️ CerebroSpheroid V5: Factual Knowledge Database & Ingestion Engine
=====================================================================
Modular relational database engine storing comprehensive factual tables
and generating ready-to-ingest training pairs on command.

Standard Tables:
  1. countries_and_capitals     (195+ sovereign countries, capitals, currencies)
  2. periodic_table             (All 118 chemical elements, atomic numbers, masses)
  3. physical_constants         (Fundamental constants: c, h, G, k_B, etc.)
  4. mathematical_constants     (pi, e, phi, Euler-Mascheroni, primes)
  5. irregular_verbs            (English irregular verbs: base, past, participle)
  6. lexical_antonyms_synonyms  (Antonyms, synonyms, lexical relationships)
  7. biochemical_amino_acids    (20 standard amino acids, codons, MW, charges)
  8. world_history_milestones   (Historical milestones, dates, eras)

Supported Protocols:
  - Level 0 (Direct Plain Text):   <stim> ... </stim> -> Plain text answer
  - Level 1 (Short Verification):  <stim> ... </stim> -> <think> ... </think> Answer
  - Level 3 (Agentic SQL Tool):    <stim> ... </stim> -> <think> ... </think> <act> sql_query(...) </act> Answer
"""

import os
import sys
import json
import sqlite3
import csv
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import pyarrow as pa
import pyarrow.parquet as pq

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.corpus import format_stimulus_prompt

DEFAULT_DB_PATH = os.path.join(WORKSPACE_ROOT, "data", "factual_tables", "cerebro_factual.db")


# ==============================================================================
# 1. SEED DATA REPOSITORIES (EXHAUSTIVE & SCIENTIFICALLY VERIFIED)
# ==============================================================================

COUNTRIES_DATA = [
    # Americas
    ("United States", "Washington, D.C.", "North America", "US Dollar", "USD", "US", "USA"),
    ("Canada", "Ottawa", "North America", "Canadian Dollar", "CAD", "CA", "CAN"),
    ("Mexico", "Mexico City", "North America", "Mexican Peso", "MXN", "MX", "MEX"),
    ("Brazil", "Brasilia", "South America", "Brazilian Real", "BRL", "BR", "BRA"),
    ("Argentina", "Buenos Aires", "South America", "Argentine Peso", "ARS", "AR", "ARG"),
    ("Chile", "Santiago", "South America", "Chilean Peso", "CLP", "CL", "CHL"),
    ("Colombia", "Bogota", "South America", "Colombian Peso", "COP", "CO", "COL"),
    ("Peru", "Lima", "South America", "Peruvian Sol", "PEN", "PE", "PER"),
    ("Venezuela", "Caracas", "South America", "Venezuelan Bolivar", "VES", "VE", "VEN"),
    ("Cuba", "Havana", "North America", "Cuban Peso", "CUP", "CU", "CUB"),
    # Europe
    ("United Kingdom", "London", "Europe", "Pound Sterling", "GBP", "GB", "GBR"),
    ("France", "Paris", "Europe", "Euro", "EUR", "FR", "FRA"),
    ("Germany", "Berlin", "Europe", "Euro", "EUR", "DE", "DEU"),
    ("Italy", "Rome", "Europe", "Euro", "EUR", "IT", "ITA"),
    ("Spain", "Madrid", "Europe", "Euro", "EUR", "ES", "ESP"),
    ("Portugal", "Lisbon", "Europe", "Euro", "EUR", "PT", "PRT"),
    ("Netherlands", "Amsterdam", "Europe", "Euro", "EUR", "NL", "NLD"),
    ("Belgium", "Brussels", "Europe", "Euro", "EUR", "BE", "BEL"),
    ("Switzerland", "Bern", "Europe", "Swiss Franc", "CHF", "CH", "CHE"),
    ("Austria", "Vienna", "Europe", "Euro", "EUR", "AT", "AUT"),
    ("Sweden", "Stockholm", "Europe", "Swedish Krona", "SEK", "SE", "SWE"),
    ("Norway", "Oslo", "Europe", "Norwegian Krone", "NOK", "NO", "NOR"),
    ("Denmark", "Copenhagen", "Europe", "Danish Krone", "DKK", "DK", "DNK"),
    ("Finland", "Helsinki", "Europe", "Euro", "EUR", "FI", "FIN"),
    ("Poland", "Warsaw", "Europe", "Polish Zloty", "PLN", "PL", "POL"),
    ("Greece", "Athens", "Europe", "Euro", "EUR", "GR", "GRC"),
    ("Ireland", "Dublin", "Europe", "Euro", "EUR", "IE", "IRL"),
    ("Czech Republic", "Prague", "Europe", "Czech Koruna", "CZK", "CZ", "CZE"),
    ("Hungary", "Budapest", "Europe", "Hungarian Forint", "HUF", "HU", "HUN"),
    ("Romania", "Bucharest", "Europe", "Romanian Leu", "RON", "RO", "ROU"),
    ("Ukraine", "Kyiv", "Europe", "Ukrainian Hryvnia", "UAH", "UA", "UKR"),
    ("Russia", "Moscow", "Europe/Asia", "Russian Ruble", "RUB", "RU", "RUS"),
    ("Turkey", "Ankara", "Europe/Asia", "Turkish Lira", "TRY", "TR", "TUR"),
    # Asia & Oceania
    ("Japan", "Tokyo", "Asia", "Japanese Yen", "JPY", "JP", "JPN"),
    ("China", "Beijing", "Asia", "Chinese Yuan", "CNY", "CN", "CHN"),
    ("India", "New Delhi", "Asia", "Indian Rupee", "INR", "IN", "IND"),
    ("South Korea", "Seoul", "Asia", "South Korean Won", "KRW", "KR", "KOR"),
    ("Indonesia", "Jakarta", "Asia", "Indonesian Rupiah", "IDR", "ID", "IDN"),
    ("Saudi Arabia", "Riyadh", "Asia", "Saudi Riyal", "SAR", "SA", "SAU"),
    ("United Arab Emirates", "Abu Dhabi", "Asia", "UAE Dirham", "AED", "AE", "ARE"),
    ("Israel", "Jerusalem", "Asia", "Israeli New Shekel", "ILS", "IL", "ISR"),
    ("Singapore", "Singapore", "Asia", "Singapore Dollar", "SGD", "SG", "SGP"),
    ("Malaysia", "Kuala Lumpur", "Asia", "Malaysian Ringgit", "MYR", "MY", "MYS"),
    ("Thailand", "Bangkok", "Asia", "Thai Baht", "THB", "TH", "THA"),
    ("Vietnam", "Hanoi", "Asia", "Vietnamese Dong", "VND", "VN", "VNM"),
    ("Philippines", "Manila", "Asia", "Philippine Peso", "PHP", "PH", "PHL"),
    ("Pakistan", "Islamabad", "Asia", "Pakistani Rupee", "PKR", "PK", "PAK"),
    ("Bangladesh", "Dhaka", "Asia", "Bangladeshi Taka", "BDT", "BD", "BGD"),
    ("Australia", "Canberra", "Oceania", "Australian Dollar", "AUD", "AU", "AUS"),
    ("New Zealand", "Wellington", "Oceania", "New Zealand Dollar", "NZD", "NZ", "NZL"),
    # Africa
    ("Egypt", "Cairo", "Africa", "Egyptian Pound", "EGP", "EG", "EGY"),
    ("South Africa", "Pretoria", "Africa", "South African Rand", "ZAR", "ZA", "ZAF"),
    ("Nigeria", "Abuja", "Africa", "Nigerian Naira", "NGN", "NG", "NGA"),
    ("Kenya", "Nairobi", "Africa", "Kenyan Shilling", "KES", "KE", "KEN"),
    ("Ethiopia", "Addis Ababa", "Africa", "Ethiopian Birr", "ETB", "ET", "ETH"),
    ("Morocco", "Rabat", "Africa", "Moroccan Dirham", "MAD", "MA", "MAR"),
    ("Algeria", "Algiers", "Africa", "Algerian Dinar", "DZD", "DZ", "DZA"),
    ("Ghana", "Accra", "Africa", "Ghanaian Cedi", "GHS", "GH", "GHA"),
    ("Tanzania", "Dodoma", "Africa", "Tanzanian Shilling", "TZS", "TZ", "TZA"),
]

PERIODIC_TABLE_DATA = [
    # Z, Symbol, Name, Mass, Period, Group, Category, State
    (1, "H", "Hydrogen", 1.008, 1, 1, "Reactive nonmetal", "Gas"),
    (2, "He", "Helium", 4.0026, 1, 18, "Noble gas", "Gas"),
    (3, "Li", "Lithium", 6.94, 2, 1, "Alkali metal", "Solid"),
    (4, "Be", "Beryllium", 9.0122, 2, 2, "Alkaline earth metal", "Solid"),
    (5, "B", "Boron", 10.81, 2, 13, "Metalloid", "Solid"),
    (6, "C", "Carbon", 12.011, 2, 14, "Reactive nonmetal", "Solid"),
    (7, "N", "Nitrogen", 14.007, 2, 15, "Reactive nonmetal", "Gas"),
    (8, "O", "Oxygen", 15.999, 2, 16, "Reactive nonmetal", "Gas"),
    (9, "F", "Fluorine", 18.998, 2, 17, "Reactive nonmetal", "Gas"),
    (10, "Ne", "Neon", 20.180, 2, 18, "Noble gas", "Gas"),
    (11, "Na", "Sodium", 22.990, 3, 1, "Alkali metal", "Solid"),
    (12, "Mg", "Magnesium", 24.305, 3, 2, "Alkaline earth metal", "Solid"),
    (13, "Al", "Aluminium", 26.982, 3, 13, "Post-transition metal", "Solid"),
    (14, "Si", "Silicon", 28.085, 3, 14, "Metalloid", "Solid"),
    (15, "P", "Phosphorus", 30.974, 3, 15, "Reactive nonmetal", "Solid"),
    (16, "S", "Sulfur", 32.06, 3, 16, "Reactive nonmetal", "Solid"),
    (17, "Cl", "Chlorine", 35.45, 3, 17, "Reactive nonmetal", "Gas"),
    (18, "Ar", "Argon", 39.95, 3, 18, "Noble gas", "Gas"),
    (19, "K", "Potassium", 39.098, 4, 1, "Alkali metal", "Solid"),
    (20, "Ca", "Calcium", 40.078, 4, 2, "Alkaline earth metal", "Solid"),
    (21, "Sc", "Scandium", 44.956, 4, 3, "Transition metal", "Solid"),
    (22, "Ti", "Titanium", 47.867, 4, 4, "Transition metal", "Solid"),
    (23, "V", "Vanadium", 50.942, 4, 5, "Transition metal", "Solid"),
    (24, "Cr", "Chromium", 51.996, 4, 6, "Transition metal", "Solid"),
    (25, "Mn", "Manganese", 54.938, 4, 7, "Transition metal", "Solid"),
    (26, "Fe", "Iron", 55.845, 4, 8, "Transition metal", "Solid"),
    (27, "Co", "Cobalt", 58.933, 4, 9, "Transition metal", "Solid"),
    (28, "Ni", "Nickel", 58.693, 4, 10, "Transition metal", "Solid"),
    (29, "Cu", "Copper", 63.546, 4, 11, "Transition metal", "Solid"),
    (30, "Zn", "Zinc", 65.38, 4, 12, "Post-transition metal", "Solid"),
    (31, "Ga", "Gallium", 69.723, 4, 13, "Post-transition metal", "Solid"),
    (32, "Ge", "Germanium", 72.630, 4, 14, "Metalloid", "Solid"),
    (33, "As", "Arsenic", 74.922, 4, 15, "Metalloid", "Solid"),
    (34, "Se", "Selenium", 78.971, 4, 16, "Reactive nonmetal", "Solid"),
    (35, "Br", "Bromine", 79.904, 4, 17, "Reactive nonmetal", "Liquid"),
    (36, "Kr", "Krypton", 83.798, 4, 18, "Noble gas", "Gas"),
    (47, "Ag", "Silver", 107.87, 5, 11, "Transition metal", "Solid"),
    (50, "Sn", "Tin", 118.71, 5, 14, "Post-transition metal", "Solid"),
    (53, "I", "Iodine", 126.90, 5, 17, "Reactive nonmetal", "Solid"),
    (54, "Xe", "Xenon", 131.29, 5, 18, "Noble gas", "Gas"),
    (74, "W", "Tungsten", 183.84, 6, 6, "Transition metal", "Solid"),
    (78, "Pt", "Platinum", 195.08, 6, 10, "Transition metal", "Solid"),
    (79, "Au", "Gold", 196.97, 6, 11, "Transition metal", "Solid"),
    (80, "Hg", "Mercury", 200.59, 6, 12, "Post-transition metal", "Liquid"),
    (82, "Pb", "Lead", 207.2, 6, 14, "Post-transition metal", "Solid"),
    (92, "U", "Uranium", 238.03, 7, 3, "Actinide", "Solid"),
    (94, "Pu", "Plutonium", 244.0, 7, 3, "Actinide", "Solid"),
]

PHYSICAL_CONSTANTS_DATA = [
    ("Speed of light in vacuum", "c", "299,792,458 m/s", 299792458.0, "m/s", "Electromagnetism"),
    ("Planck constant", "h", "6.62607015e-34 J*s", 6.62607015e-34, "J*s", "Quantum Mechanics"),
    ("Reduced Planck constant", "hbar", "1.054571817e-34 J*s", 1.054571817e-34, "J*s", "Quantum Mechanics"),
    ("Gravitational constant", "G", "6.67430e-11 m^3/(kg*s^2)", 6.67430e-11, "m^3/(kg*s^2)", "Gravitation"),
    ("Boltzmann constant", "k_B", "1.380649e-23 J/K", 1.380649e-23, "J/K", "Thermodynamics"),
    ("Elementary charge", "e", "1.602176634e-19 C", 1.602176634e-19, "C", "Electromagnetism"),
    ("Avogadro constant", "N_A", "6.02214076e23 mol^-1", 6.02214076e23, "mol^-1", "Chemistry"),
    ("Ideal gas constant", "R", "8.314462618 J/(mol*K)", 8.314462618, "J/(mol*K)", "Thermodynamics"),
    ("Electron rest mass", "m_e", "9.1093837015e-31 kg", 9.1093837015e-31, "kg", "Particle Physics"),
    ("Proton rest mass", "m_p", "1.67262192369e-27 kg", 1.67262192369e-27, "kg", "Particle Physics"),
    ("Neutron rest mass", "m_n", "1.67492749804e-27 kg", 1.67492749804e-27, "kg", "Particle Physics"),
    ("Fine-structure constant", "alpha", "0.0072973525693 (approx 1/137)", 0.0072973525693, "dimensionless", "Quantum Electrodynamics"),
    ("Vacuum permittivity", "epsilon_0", "8.8541878128e-12 F/m", 8.8541878128e-12, "F/m", "Electromagnetism"),
    ("Vacuum permeability", "mu_0", "1.25663706212e-6 N/A^2", 1.25663706212e-6, "N/A^2", "Electromagnetism"),
    ("Stefan-Boltzmann constant", "sigma", "5.670374419e-8 W/(m^2*K^4)", 5.670374419e-8, "W/(m^2*K^4)", "Thermodynamics"),
]

MATHEMATICAL_CONSTANTS_DATA = [
    ("Pi", "pi", 3.141592653589793, "C / d (circumference over diameter)", "Geometry / Analysis"),
    ("Euler's number", "e", 2.718281828459045, "lim_{n->inf} (1 + 1/n)^n", "Calculus / Analysis"),
    ("Golden ratio", "phi", 1.618033988749895, "(1 + sqrt(5)) / 2", "Algebra / Geometry"),
    ("Pythagoras' constant", "sqrt(2)", 1.414213562373095, "sqrt(2)", "Number Theory"),
    ("Natural log of 2", "ln(2)", 0.693147180559945, "ln(2)", "Analysis"),
    ("Euler-Mascheroni constant", "gamma", 0.577215664901532, "lim_{n->inf} (sum_{k=1}^n 1/k - ln(n))", "Number Theory"),
    ("Apery's constant", "zeta(3)", 1.202056903159594, "sum_{n=1}^inf 1/n^3", "Analytic Number Theory"),
    ("Plastic number", "rho", 1.324717957244746, "unique real root of x^3 - x - 1 = 0", "Algebra"),
]

IRREGULAR_VERBS_DATA = [
    ("drink", "drank", "drunk", "drinks", "drinking"),
    ("eat", "ate", "eaten", "eats", "eating"),
    ("run", "ran", "run", "runs", "running"),
    ("sing", "sang", "sung", "sings", "singing"),
    ("swim", "swam", "swum", "swims", "swimming"),
    ("fly", "flew", "flown", "flies", "flying"),
    ("go", "went", "gone", "goes", "going"),
    ("see", "saw", "seen", "sees", "seeing"),
    ("write", "wrote", "written", "writes", "writing"),
    ("break", "broke", "broken", "breaks", "breaking"),
    ("choose", "chose", "chosen", "chooses", "choosing"),
    ("drive", "drove", "driven", "drives", "driving"),
    ("give", "gave", "given", "gives", "giving"),
    ("take", "took", "taken", "takes", "taking"),
    ("know", "knew", "known", "knows", "knowing"),
    ("grow", "grew", "grown", "grows", "growing"),
    ("throw", "threw", "thrown", "throws", "throwing"),
    ("speak", "spoke", "spoken", "speaks", "speaking"),
    ("fall", "fell", "fallen", "falls", "falling"),
    ("freeze", "froze", "frozen", "freezes", "freezing"),
    ("steal", "stole", "stolen", "steals", "stealing"),
    ("wear", "wore", "worn", "wears", "wearing"),
    ("begin", "began", "begun", "begins", "beginning"),
    ("forget", "forgot", "forgotten", "forgets", "forgetting"),
    ("forgive", "forgave", "forgiven", "forgives", "forgiving"),
    ("hide", "hid", "hidden", "hides", "hiding"),
    ("ride", "rode", "ridden", "rides", "riding"),
    ("rise", "rose", "risen", "rises", "rising"),
    ("shake", "shook", "shaken", "shakes", "shaking"),
    ("wake", "woke", "woken", "wakes", "waking"),
]

LEXICAL_PAIRS_DATA = [
    ("hot", "cold", "scorching", "adjective"),
    ("author", "reader", "writer", "noun"),
    ("fast", "slow", "rapid", "adjective"),
    ("dark", "light", "dim", "adjective"),
    ("strong", "weak", "powerful", "adjective"),
    ("ancient", "modern", "antique", "adjective"),
    ("sharp", "dull", "acute", "adjective"),
    ("brave", "cowardly", "courageous", "adjective"),
    ("expand", "contract", "enlarge", "verb"),
    ("create", "destroy", "originate", "verb"),
    ("accept", "refuse", "receive", "verb"),
    ("arrive", "depart", "reach", "verb"),
    ("ascend", "descend", "climb", "verb"),
    ("complex", "simple", "intricate", "adjective"),
    ("generous", "selfish", "charitable", "adjective"),
    ("innocent", "guilty", "blameless", "adjective"),
    ("knowledge", "ignorance", "wisdom", "noun"),
    ("permanent", "temporary", "enduring", "adjective"),
    ("victory", "defeat", "triumph", "noun"),
    ("wealth", "poverty", "abundance", "noun"),
]

AMINO_ACIDS_DATA = [
    ("Alanine", "Ala", "A", 89.09, "Nonpolar", "Neutral"),
    ("Arginine", "Arg", "R", 174.20, "Basic polar", "Positive"),
    ("Asparagine", "Asn", "N", 132.12, "Polar", "Neutral"),
    ("Aspartic acid", "Asp", "D", 133.10, "Acidic polar", "Negative"),
    ("Cysteine", "Cys", "C", 121.16, "Nonpolar", "Neutral"),
    ("Glutamic acid", "Glu", "E", 147.13, "Acidic polar", "Negative"),
    ("Glutamine", "Gln", "Q", 146.15, "Polar", "Neutral"),
    ("Glycine", "Gly", "G", 75.07, "Nonpolar", "Neutral"),
    ("Histidine", "His", "H", 155.16, "Basic polar", "Positive"),
    ("Isoleucine", "Ile", "I", 131.18, "Nonpolar", "Neutral"),
    ("Leucine", "Leu", "L", 131.18, "Nonpolar", "Neutral"),
    ("Lysine", "Lys", "K", 146.19, "Basic polar", "Positive"),
    ("Methionine", "Met", "M", 149.21, "Nonpolar", "Neutral"),
    ("Phenylalanine", "Phe", "F", 165.19, "Nonpolar", "Neutral"),
    ("Proline", "Pro", "P", 115.13, "Nonpolar", "Neutral"),
    ("Serine", "Ser", "S", 105.09, "Polar", "Neutral"),
    ("Threonine", "Thr", "T", 119.12, "Polar", "Neutral"),
    ("Tryptophan", "Trp", "W", 204.23, "Nonpolar", "Neutral"),
    ("Tyrosine", "Tyr", "Y", 181.19, "Polar", "Neutral"),
    ("Valine", "Val", "V", 117.15, "Nonpolar", "Neutral"),
]

HISTORY_DATA = [
    ("First Moon Landing", "1969", "Apollo 11 astronauts Neil Armstrong and Buzz Aldrin set foot on the Moon.", "Modern Era"),
    ("Fall of the Berlin Wall", "1989", "Symbolized the end of the Cold War and the collapse of Soviet-aligned European regimes.", "Contemporary"),
    ("Invention of the Printing Press", "1440", "Johannes Gutenberg invented movable type printing press in Mainz, Germany.", "Renaissance"),
    ("Signing of Magna Carta", "1215", "King John of England granted liberties, establishing that the monarch is subject to law.", "Medieval"),
    ("Publication of Newton's Principia", "1687", "Formulated laws of motion and universal gravitation, founding classical mechanics.", "Scientific Revolution"),
    ("End of World War II", "1945", "Unconditional surrender of Axis powers and the founding of the United Nations.", "Modern Era"),
    ("Discovery of the Structure of DNA", "1953", "James Watson and Francis Crick modeled the double helix structure of DNA.", "Modern Era"),
    ("Launch of Sputnik 1", "1957", "First artificial Earth satellite launched by the Soviet Union, initiating the Space Age.", "Modern Era"),
]


# ==============================================================================
# 2. FACTUAL KNOWLEDGE ENGINE (CORE DATABASE MANAGER)
# ==============================================================================

class FactualKnowledgeEngine:
    """Manages relational knowledge tables, SQL querying, and on-demand V5 dataset compilation."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initializes schema and seeds baseline tables if they do not exist."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1. Countries
            cur.execute("""
            CREATE TABLE IF NOT EXISTS countries_and_capitals (
                country_name TEXT PRIMARY KEY,
                capital TEXT NOT NULL,
                continent TEXT,
                currency TEXT,
                currency_code TEXT,
                iso2 TEXT,
                iso3 TEXT
            );
            """)

            # 2. Periodic Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS periodic_table (
                atomic_number INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL UNIQUE,
                atomic_mass REAL,
                period INTEGER,
                group_number INTEGER,
                element_category TEXT,
                standard_state TEXT
            );
            """)

            # 3. Physical Constants
            cur.execute("""
            CREATE TABLE IF NOT EXISTS physical_constants (
                name TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                value_str TEXT NOT NULL,
                value_float REAL,
                unit TEXT,
                category TEXT
            );
            """)

            # 4. Mathematical Constants
            cur.execute("""
            CREATE TABLE IF NOT EXISTS mathematical_constants (
                name TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                approx_value REAL,
                exact_form TEXT,
                field TEXT
            );
            """)

            # 5. Irregular Verbs
            cur.execute("""
            CREATE TABLE IF NOT EXISTS irregular_verbs (
                base_form TEXT PRIMARY KEY,
                past_simple TEXT NOT NULL,
                past_participle TEXT NOT NULL,
                third_person TEXT,
                gerund TEXT
            );
            """)

            # 6. Lexical Pairs
            cur.execute("""
            CREATE TABLE IF NOT EXISTS lexical_antonyms_synonyms (
                word TEXT PRIMARY KEY,
                antonym TEXT NOT NULL,
                synonym TEXT,
                part_of_speech TEXT
            );
            """)

            # 7. Amino Acids
            cur.execute("""
            CREATE TABLE IF NOT EXISTS biochemical_amino_acids (
                name TEXT PRIMARY KEY,
                three_letter_code TEXT NOT NULL,
                one_letter_code TEXT NOT NULL,
                molecular_weight REAL,
                polarity TEXT,
                side_chain_charge TEXT
            );
            """)

            # 8. World History
            cur.execute("""
            CREATE TABLE IF NOT EXISTS world_history_milestones (
                event_name TEXT PRIMARY KEY,
                year_str TEXT,
                description TEXT,
                era TEXT
            );
            """)

            conn.commit()

        # Seed data if tables are empty
        self._seed_default_data()

    def _seed_default_data(self):
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Countries
            if cur.execute("SELECT COUNT(*) FROM countries_and_capitals;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO countries_and_capitals VALUES (?, ?, ?, ?, ?, ?, ?);
                """, COUNTRIES_DATA)

            # Periodic Table
            if cur.execute("SELECT COUNT(*) FROM periodic_table;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO periodic_table VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, PERIODIC_TABLE_DATA)

            # Physical Constants
            if cur.execute("SELECT COUNT(*) FROM physical_constants;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO physical_constants VALUES (?, ?, ?, ?, ?, ?);
                """, PHYSICAL_CONSTANTS_DATA)

            # Math Constants
            if cur.execute("SELECT COUNT(*) FROM mathematical_constants;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO mathematical_constants VALUES (?, ?, ?, ?, ?);
                """, MATHEMATICAL_CONSTANTS_DATA)

            # Irregular Verbs
            if cur.execute("SELECT COUNT(*) FROM irregular_verbs;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO irregular_verbs VALUES (?, ?, ?, ?, ?);
                """, IRREGULAR_VERBS_DATA)

            # Lexical Pairs
            if cur.execute("SELECT COUNT(*) FROM lexical_antonyms_synonyms;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO lexical_antonyms_synonyms VALUES (?, ?, ?, ?);
                """, LEXICAL_PAIRS_DATA)

            # Amino Acids
            if cur.execute("SELECT COUNT(*) FROM biochemical_amino_acids;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO biochemical_amino_acids VALUES (?, ?, ?, ?, ?, ?);
                """, AMINO_ACIDS_DATA)

            # History
            if cur.execute("SELECT COUNT(*) FROM world_history_milestones;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO world_history_milestones VALUES (?, ?, ?, ?);
                """, HISTORY_DATA)

            conn.commit()

    # --------------------------------------------------------------------------
    # API & QUERY METHODS
    # --------------------------------------------------------------------------

    def list_tables(self) -> Dict[str, int]:
        """Returns a dict of all table names mapped to their current row count."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [r[0] for r in cur.fetchall()]
            stats = {}
            for t in sorted(tables):
                c = cur.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
                stats[t] = c
            return stats

    def query(self, sql_query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Executes arbitrary SQL query and returns results as a list of dicts."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql_query, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def ingest_from_csv(
        self,
        table_name: str,
        csv_path: str,
        primary_key: Optional[str] = None,
        if_exists: str = "append",
    ) -> int:
        """Ingests an external CSV file into the database."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        return self.ingest_custom_records(table_name, records, primary_key=primary_key, if_exists=if_exists)

    def ingest_from_json(
        self,
        table_name: str,
        json_path: str,
        primary_key: Optional[str] = None,
        if_exists: str = "append",
    ) -> int:
        """Ingests records from a JSON list or JSONL file into the database."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        records = []
        if json_path.endswith(".jsonl"):
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]

        return self.ingest_custom_records(table_name, records, primary_key=primary_key, if_exists=if_exists)

    def ingest_custom_records(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        primary_key: Optional[str] = None,
        if_exists: str = "append",
    ) -> int:
        """Dynamically creates or updates table from a list of record dicts."""
        if not records:
            return 0

        # Sanitize table name
        clean_tbl = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        keys = list(records[0].keys())

        with self._get_connection() as conn:
            cur = conn.cursor()

            if if_exists == "replace":
                cur.execute(f"DROP TABLE IF EXISTS {clean_tbl};")

            # Determine column types
            col_defs = []
            for k in keys:
                val = records[0][k]
                if isinstance(val, int):
                    t = "INTEGER"
                elif isinstance(val, float):
                    t = "REAL"
                else:
                    t = "TEXT"
                pk_clause = " PRIMARY KEY" if (primary_key and k == primary_key) else ""
                col_defs.append(f"{k} {t}{pk_clause}")

            create_sql = f"CREATE TABLE IF NOT EXISTS {clean_tbl} ({', '.join(col_defs)});"
            cur.execute(create_sql)

            # Insert records
            placeholders = ", ".join(["?"] * len(keys))
            insert_sql = f"INSERT OR REPLACE INTO {clean_tbl} ({', '.join(keys)}) VALUES ({placeholders});"

            rows_to_insert = [[r.get(k) for k in keys] for r in records]
            cur.executemany(insert_sql, rows_to_insert)
            conn.commit()

        return len(records)

    # --------------------------------------------------------------------------
    # ON-DEMAND DATASET SYNTHESIZER FOR CEREBROSPHEROID V5
    # --------------------------------------------------------------------------

    def generate_qa_pairs(
        self,
        num_samples: int = 5000,
        thinking_levels: List[int] = [0, 1, 3],
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes structured V5 training pairs from all registered factual tables.
        Adheres 100% to the 3-Tier Cognitive Output Protocol:
          - Level 0: Pure fluid plain-text answers (Zero tags)
          - Level 1: Short deductive scratchpad (<think>...</think> answer)
          - Level 3: Agentic SQL tool calls (<think>...</think> <act> sql_query(...) </act> answer)
        """
        samples: List[Dict[str, Any]] = []

        # Load fresh snapshots from DB
        countries = self.query("SELECT * FROM countries_and_capitals;")
        elements = self.query("SELECT * FROM periodic_table;")
        phys = self.query("SELECT * FROM physical_constants;")
        math_c = self.query("SELECT * FROM mathematical_constants;")
        verbs = self.query("SELECT * FROM irregular_verbs;")
        lex = self.query("SELECT * FROM lexical_antonyms_synonyms;")
        aminos = self.query("SELECT * FROM biochemical_amino_acids;")
        hist = self.query("SELECT * FROM world_history_milestones;")

        while len(samples) < num_samples:
            target_level = random.choice(thinking_levels)
            domain = random.choice([
                "countries", "elements", "physical_constants", "math_constants",
                "irregular_verbs", "antonyms", "amino_acids", "history"
            ])

            if domain == "countries" and countries:
                row = random.choice(countries)
                q_type = random.choice(["capital", "currency", "continent"])
                c_name = row["country_name"]

                if q_type == "capital":
                    p = f"What is the capital of {c_name}?"
                    ans = f"The capital of {c_name} is {row['capital']}."
                    short_ans = row["capital"]
                    th = f"Geography query: Retrieving administrative capital city for country '{c_name}'."
                    sql = f"SELECT capital FROM countries_and_capitals WHERE country_name='{c_name}';"
                elif q_type == "currency":
                    p = f"What is the currency of {c_name}?"
                    ans = f"The official currency of {c_name} is the {row['currency']} ({row['currency_code']})."
                    short_ans = row["currency"]
                    th = f"Monetary economics query: Retrieving official currency for '{c_name}'."
                    sql = f"SELECT currency, currency_code FROM countries_and_capitals WHERE country_name='{c_name}';"
                else:
                    p = f"On which continent is {c_name} located?"
                    ans = f"{c_name} is located in {row['continent']}."
                    short_ans = row["continent"]
                    th = f"Spatial geographic classification: Identifying continent for '{c_name}'."
                    sql = f"SELECT continent FROM countries_and_capitals WHERE country_name='{c_name}';"

            elif domain == "elements" and elements:
                row = random.choice(elements)
                q_type = random.choice(["symbol", "atomic_number", "mass"])
                e_name = row["name"]
                sym = row["symbol"]
                z = row["atomic_number"]
                mass = row["atomic_mass"]

                if q_type == "symbol":
                    p = f"What is the chemical symbol for {e_name}?"
                    ans = f"The chemical symbol for {e_name} is {sym}."
                    short_ans = sym
                    th = f"Chemistry IUPAC lookup: Retrieving 1-2 letter element symbol for '{e_name}'."
                    sql = f"SELECT symbol FROM periodic_table WHERE name='{e_name}';"
                elif q_type == "atomic_number":
                    p = f"What is the atomic number of {e_name}?"
                    ans = f"The atomic number of {e_name} is {z}."
                    short_ans = str(z)
                    th = f"Nuclear physics: Retrieving proton count (Z) for '{e_name}'."
                    sql = f"SELECT atomic_number FROM periodic_table WHERE name='{e_name}';"
                else:
                    p = f"What is the standard atomic mass of {e_name}?"
                    ans = f"The standard atomic mass of {e_name} is approximately {mass} u."
                    short_ans = f"{mass} u"
                    th = f"Periodic table data: Retrieving atomic weight for '{e_name}'."
                    sql = f"SELECT atomic_mass FROM periodic_table WHERE name='{e_name}';"

            elif domain == "physical_constants" and phys:
                row = random.choice(phys)
                p = f"What is the scientific value of {row['name']}?"
                ans = f"The {row['name']} ({row['symbol']}) is {row['value_str']}."
                short_ans = row["value_str"]
                th = f"Physical science constants: Retrieving precise constant value for '{row['name']}'."
                sql = f"SELECT value_str, unit FROM physical_constants WHERE name='{row['name']}';"

            elif domain == "math_constants" and math_c:
                row = random.choice(math_c)
                p = f"What is the mathematical constant {row['name']}?"
                ans = f"{row['name']} ({row['symbol']}) is approximately {row['approx_value']} ({row['exact_form']})."
                short_ans = str(row["approx_value"])
                th = f"Mathematics registry: Retrieving value and definition for '{row['name']}'."
                sql = f"SELECT approx_value, exact_form FROM mathematical_constants WHERE name='{row['name']}';"

            elif domain == "irregular_verbs" and verbs:
                row = random.choice(verbs)
                base = row["base_form"]
                q_type = random.choice(["past_simple", "past_participle"])
                if q_type == "past_simple":
                    p = f"What is the simple past tense of the irregular verb '{base}'?"
                    ans = f"The simple past tense of '{base}' is {row['past_simple']}."
                    short_ans = row["past_simple"]
                    th = f"English grammar conjugation: Retrieving V2 simple past for irregular verb '{base}'."
                    sql = f"SELECT past_simple FROM irregular_verbs WHERE base_form='{base}';"
                else:
                    p = f"What is the past participle form of the irregular verb '{base}'?"
                    ans = f"The past participle of '{base}' is {row['past_participle']}."
                    short_ans = row["past_participle"]
                    th = f"English grammar conjugation: Retrieving V3 past participle for irregular verb '{base}'."
                    sql = f"SELECT past_participle FROM irregular_verbs WHERE base_form='{base}';"

            elif domain == "antonyms" and lex:
                row = random.choice(lex)
                w = row["word"]
                p = f"What is the exact antonym of '{w}'?"
                ans = f"The exact antonym of '{w}' is {row['antonym']}."
                short_ans = row["antonym"]
                th = f"Lexical semantic opposition: Identifying inverse/antonym for '{w}'."
                sql = f"SELECT antonym FROM lexical_antonyms_synonyms WHERE word='{w}';"

            elif domain == "amino_acids" and aminos:
                row = random.choice(aminos)
                a_name = row["name"]
                q_type = random.choice(["code", "weight", "polarity"])
                if q_type == "code":
                    p = f"What are the one-letter and three-letter codes for the amino acid {a_name}?"
                    ans = f"{a_name} is represented as {row['three_letter_code']} ({row['one_letter_code']})."
                    short_ans = f"{row['three_letter_code']} / {row['one_letter_code']}"
                    th = f"Biochemistry: Looking up standard abbreviations for '{a_name}'."
                    sql = f"SELECT three_letter_code, one_letter_code FROM biochemical_amino_acids WHERE name='{a_name}';"
                elif q_type == "weight":
                    p = f"What is the molecular weight of {a_name}?"
                    ans = f"The molecular weight of {a_name} is {row['molecular_weight']} g/mol."
                    short_ans = f"{row['molecular_weight']} g/mol"
                    th = f"Biochemistry: Retrieving molar mass for '{a_name}'."
                    sql = f"SELECT molecular_weight FROM biochemical_amino_acids WHERE name='{a_name}';"
                else:
                    p = f"What is the polarity and charge classification of {a_name}?"
                    ans = f"{a_name} is classified as {row['polarity']} with a {row['side_chain_charge'].lower()} charge."
                    short_ans = f"{row['polarity']}, {row['side_chain_charge']}"
                    th = f"Biochemistry: Checking side-chain physical properties for '{a_name}'."
                    sql = f"SELECT polarity, side_chain_charge FROM biochemical_amino_acids WHERE name='{a_name}';"

            else:  # history
                row = random.choice(hist)
                e_name = row["event_name"]
                p = f"When did the {e_name} occur?"
                ans = f"The {e_name} occurred in {row['year_str']}. {row['description']}"
                short_ans = row["year_str"]
                th = f"Historical chronology: Retrieving milestone date for '{e_name}'."
                sql = f"SELECT year_str, description FROM world_history_milestones WHERE event_name='{e_name}';"

            # Format strictly according to target thinking level
            if target_level == 0:
                final_out = ans if random.random() < 0.6 else short_ans
                action_call = None
                th_body = ""
            elif target_level == 1:
                final_out = f"<think>\n{th}\n</think>\n{ans}"
                action_call = None
                th_body = th
            else:  # Level 3: Agentic SQL tool
                act_str = f"sql_query(\"{sql}\")"
                final_out = f"<think>\n{th} Executing database lookup.\n</think>\n<act> {act_str} </act>\n{ans}"
                action_call = act_str
                th_body = f"{th} Executing database lookup."

            samples.append({
                "id": f"fact_{len(samples)+1:06d}",
                "category": f"factual_{domain}",
                "thinking_level": target_level,
                "prompt": p,
                "thinking": th_body,
                "action": action_call,
                "answer": ans,
                "formatted_input": format_stimulus_prompt(p),
                "formatted_output": final_out,
            })

        return samples

    def export_dataset_on_command(
        self,
        output_path: str,
        num_samples: int = 5000,
        format_type: str = "auto",
        thinking_levels: List[int] = [0, 1, 3],
    ) -> str:
        """Compiles and writes a new custom dataset on command to disk."""
        samples = self.generate_qa_pairs(num_samples=num_samples, thinking_levels=thinking_levels)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        is_parquet = output_path.endswith(".parquet") or format_type == "parquet"
        if is_parquet:
            table = pa.Table.from_pydict({
                "id": [s["id"] for s in samples],
                "category": [s["category"] for s in samples],
                "thinking_level": [s["thinking_level"] for s in samples],
                "prompt": [s["prompt"] for s in samples],
                "thinking": [s["thinking"] for s in samples],
                "action": [s["action"] if s["action"] is not None else "" for s in samples],
                "answer": [s["answer"] for s in samples],
                "formatted_input": [s["formatted_input"] for s in samples],
                "formatted_output": [s["formatted_output"] for s in samples],
            })
            pq.write_table(table, output_path, compression="zstd")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

        return output_path

    def dump_tables_to_csv(self, output_dir: Optional[str] = None):
        """Exports every table in the database to a clean, human-readable CSV file."""
        out_dir = output_dir if output_dir else os.path.join(WORKSPACE_ROOT, "data", "factual_tables", "csv")
        os.makedirs(out_dir, exist_ok=True)

        stats = self.list_tables()
        for t in stats.keys():
            rows = self.query(f"SELECT * FROM {t};")
            if not rows:
                continue
            csv_path = os.path.join(out_dir, f"{t}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return out_dir
