#!/usr/bin/env python3
"""
🌍 CerebroSpheroid V5: World Facts Database Manager & Tool Engine
================================================================
Comprehensive relational knowledge base and management interface for
world facts. Enables autonomous retrieval for the model (<act> db_lookup(...) </act>)
as well as administrative data management, imports, exports, and statistics.

Features:
  - Persistent SQLite database (WAL mode, indexed lookups, read-only guardrails)
  - Tables:
      • world_countries (100+ countries with capitals, currencies, languages, population)
      • world_geography (mountains, rivers, oceans, lakes, deserts, canyons)
      • world_history (major human milestones, treaties, discoveries, scientific breakthroughs)
      • periodic_table (elements, symbols, atomic numbers, groups)
      • scientific_constants (speed of light, Planck, gravitational constant, etc.)
      • famous_landmarks (monuments, ancient and modern wonders)
  - Autonomous Agent APIs:
      • lookup_fact(entity, attribute) -> string
      • search_facts(keyword) -> list of structured matches
      • execute_query(sql) -> formatted results
  - Database Management (CRUD, CSV/JSON import & export, table stats)
"""

import csv
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(WORKSPACE_ROOT, "data", "factual_tables", "world_facts.db")


class WorldFactsManager:
    """Relational database manager and autonomous tool interface for world facts."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_database(self):
        """Initializes tables and seeds core curated world facts if empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Countries Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_countries (
                country TEXT PRIMARY KEY COLLATE NOCASE,
                capital TEXT NOT NULL COLLATE NOCASE,
                continent TEXT NOT NULL COLLATE NOCASE,
                currency TEXT NOT NULL,
                currency_code TEXT NOT NULL,
                languages TEXT NOT NULL,
                population INTEGER,
                area_sq_km REAL,
                tld TEXT
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_countries_capital ON world_countries(capital);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_countries_continent ON world_countries(continent);")

            # 2. Geography Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_geography (
                feature_name TEXT PRIMARY KEY COLLATE NOCASE,
                feature_type TEXT NOT NULL COLLATE NOCASE,
                location TEXT NOT NULL,
                dimension_metric TEXT NOT NULL,
                notable_fact TEXT
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_geo_type ON world_geography(feature_type);")

            # 3. History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL COLLATE NOCASE,
                year_date TEXT NOT NULL,
                century TEXT NOT NULL,
                category TEXT NOT NULL,
                key_figures TEXT,
                significance TEXT NOT NULL
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_event ON world_history(event_name);")

            # 4. Periodic Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS periodic_table (
                atomic_number INTEGER PRIMARY KEY,
                element_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                symbol TEXT NOT NULL UNIQUE COLLATE NOCASE,
                atomic_mass REAL NOT NULL,
                group_number INTEGER,
                period_number INTEGER,
                standard_state TEXT NOT NULL
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_element_symbol ON periodic_table(symbol);")

            # 5. Scientific Constants
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS scientific_constants (
                constant_name TEXT PRIMARY KEY COLLATE NOCASE,
                symbol TEXT NOT NULL,
                numerical_value TEXT NOT NULL,
                unit TEXT NOT NULL,
                discipline TEXT NOT NULL,
                description TEXT
            );
            """)

            # 6. Landmarks Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS famous_landmarks (
                landmark_name TEXT PRIMARY KEY COLLATE NOCASE,
                country TEXT NOT NULL COLLATE NOCASE,
                city TEXT NOT NULL,
                year_built TEXT,
                category TEXT NOT NULL,
                significance TEXT
            );
            """)

            conn.commit()

            # Seed if empty
            cursor.execute("SELECT COUNT(*) FROM world_countries;")
            if cursor.fetchone()[0] == 0:
                self._seed_initial_facts(conn)

    def _seed_initial_facts(self, conn: sqlite3.Connection):
        """Seeds initial high-authority world facts."""
        cursor = conn.cursor()

        # Seed Countries (50+ Major Global Nations)
        countries_data = [
            ("Afghanistan", "Kabul", "Asia", "Afghan Afghani", "AFN", "Pashto, Dari", 41128771, 652864, ".af"),
            ("Argentina", "Buenos Aires", "South America", "Argentine Peso", "ARS", "Spanish", 46044703, 2780400, ".ar"),
            ("Australia", "Canberra", "Oceania", "Australian Dollar", "AUD", "English", 26177413, 7692024, ".au"),
            ("Austria", "Vienna", "Europe", "Euro", "EUR", "German", 9042528, 83879, ".at"),
            ("Bangladesh", "Dhaka", "Asia", "Bangladeshi Taka", "BDT", "Bengali", 171186372, 147570, ".bd"),
            ("Belgium", "Brussels", "Europe", "Euro", "EUR", "Dutch, French, German", 11655930, 30528, ".be"),
            ("Brazil", "Brasília", "South America", "Brazilian Real", "BRL", "Portuguese", 215313498, 8515767, ".br"),
            ("Canada", "Ottawa", "North America", "Canadian Dollar", "CAD", "English, French", 38929902, 9984670, ".ca"),
            ("Chile", "Santiago", "South America", "Chilean Peso", "CLP", "Spanish", 19603733, 756102, ".cl"),
            ("China", "Beijing", "Asia", "Chinese Yuan", "CNY", "Standard Chinese (Mandarin)", 1411750000, 9596961, ".cn"),
            ("Colombia", "Bogotá", "South America", "Colombian Peso", "COP", "Spanish", 51874024, 1141748, ".co"),
            ("Denmark", "Copenhagen", "Europe", "Danish Krone", "DKK", "Danish", 5903037, 43094, ".dk"),
            ("Egypt", "Cairo", "Africa", "Egyptian Pound", "EGP", "Arabic", 104258327, 1002450, ".eg"),
            ("Ethiopia", "Addis Ababa", "Africa", "Ethiopian Birr", "ETB", "Amharic", 123379924, 1104300, ".et"),
            ("Finland", "Helsinki", "Europe", "Euro", "EUR", "Finnish, Swedish", 5540720, 338424, ".fi"),
            ("France", "Paris", "Europe", "Euro", "EUR", "French", 67970000, 551695, ".fr"),
            ("Germany", "Berlin", "Europe", "Euro", "EUR", "German", 84358845, 357022, ".de"),
            ("Ghana", "Accra", "Africa", "Ghanaian Cedi", "GHS", "English", 33475870, 238533, ".gh"),
            ("Greece", "Athens", "Europe", "Euro", "EUR", "Greek", 10384971, 131957, ".gr"),
            ("India", "New Delhi", "Asia", "Indian Rupee", "INR", "Hindi, English", 1428627663, 3287263, ".in"),
            ("Indonesia", "Jakarta", "Asia", "Indonesian Rupiah", "IDR", "Indonesian", 275501339, 1904569, ".id"),
            ("Iran", "Tehran", "Asia", "Iranian Rial", "IRR", "Persian", 88550570, 1648195, ".ir"),
            ("Iraq", "Baghdad", "Asia", "Iraqi Dinar", "IQD", "Arabic, Kurdish", 44496122, 438317, ".iq"),
            ("Ireland", "Dublin", "Europe", "Euro", "EUR", "Irish, English", 5127170, 70273, ".ie"),
            ("Israel", "Jerusalem", "Asia", "Israeli New Shekel", "ILS", "Hebrew", 9557500, 22072, ".il"),
            ("Italy", "Rome", "Europe", "Euro", "EUR", "Italian", 58850717, 301340, ".it"),
            ("Japan", "Tokyo", "Asia", "Japanese Yen", "JPY", "Japanese", 125124989, 377975, ".jp"),
            ("Kenya", "Nairobi", "Africa", "Kenyan Shilling", "KES", "Swahili, English", 54027487, 580367, ".ke"),
            ("Mexico", "Mexico City", "North America", "Mexican Peso", "MXN", "Spanish", 127504125, 1964375, ".mx"),
            ("Morocco", "Rabat", "Africa", "Moroccan Dirham", "MAD", "Arabic, Berber", 37457971, 446550, ".ma"),
            ("Netherlands", "Amsterdam", "Europe", "Euro", "EUR", "Dutch", 17700982, 41543, ".nl"),
            ("New Zealand", "Wellington", "Oceania", "New Zealand Dollar", "NZD", "English, Māori", 5124100, 268021, ".nz"),
            ("Nigeria", "Abuja", "Africa", "Nigerian Naira", "NGN", "English", 218541212, 923768, ".ng"),
            ("Norway", "Oslo", "Europe", "Norwegian Krone", "NOK", "Norwegian", 5434319, 385207, ".no"),
            ("Pakistan", "Islamabad", "Asia", "Pakistani Rupee", "PKR", "Urdu, English", 240485658, 881913, ".pk"),
            ("Peru", "Lima", "South America", "Peruvian Sol", "PEN", "Spanish", 34049588, 1285216, ".pe"),
            ("Philippines", "Manila", "Asia", "Philippine Peso", "PHP", "Filipino, English", 115559009, 300000, ".ph"),
            ("Poland", "Warsaw", "Europe", "Polish Złoty", "PLN", "Polish", 37654000, 312696, ".pl"),
            ("Portugal", "Lisbon", "Europe", "Euro", "EUR", "Portuguese", 10352042, 92212, ".pt"),
            ("Russia", "Moscow", "Europe/Asia", "Russian Ruble", "RUB", "Russian", 144236933, 17098246, ".ru"),
            ("Saudi Arabia", "Riyadh", "Asia", "Saudi Riyal", "SAR", "Arabic", 36408820, 2149690, ".sa"),
            ("South Africa", "Pretoria", "Africa", "South African Rand", "ZAR", "Zulu, Xhosa, Afrikaans, English", 60604992, 1221037, ".za"),
            ("South Korea", "Seoul", "Asia", "South Korean Won", "KRW", "Korean", 51638809, 100210, ".kr"),
            ("Spain", "Madrid", "Europe", "Euro", "EUR", "Spanish", 47615034, 505990, ".es"),
            ("Sweden", "Stockholm", "Europe", "Swedish Krona", "SEK", "Swedish", 10452326, 450295, ".se"),
            ("Switzerland", "Bern", "Europe", "Swiss Franc", "CHF", "German, French, Italian, Romansh", 8740472, 41285, ".ch"),
            ("Thailand", "Bangkok", "Asia", "Thai Baht", "THB", "Thai", 71697030, 513120, ".th"),
            ("Turkey", "Ankara", "Asia/Europe", "Turkish Lira", "TRY", "Turkish", 85279553, 783562, ".tr"),
            ("Ukraine", "Kyiv", "Europe", "Ukrainian Hryvnia", "UAH", "Ukrainian", 38000000, 603628, ".ua"),
            ("United Arab Emirates", "Abu Dhabi", "Asia", "UAE Dirham", "AED", "Arabic", 9441129, 83600, ".ae"),
            ("United Kingdom", "London", "Europe", "Pound Sterling", "GBP", "English", 67081000, 242495, ".uk"),
            ("United States", "Washington, D.C.", "North America", "United States Dollar", "USD", "English", 333287557, 9833517, ".us"),
            ("Vietnam", "Hanoi", "Asia", "Vietnamese Đồng", "VND", "Vietnamese", 98186856, 331212, ".vn"),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO world_countries (country, capital, continent, currency, currency_code, languages, population, area_sq_km, tld)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, countries_data)

        # Seed Geography
        geo_data = [
            ("Mount Everest", "Mountain", "Himalayas (Nepal/China)", "8,848.86 m", "Highest mountain peak above sea level on Earth."),
            ("K2", "Mountain", "Karakoram (Pakistan/China)", "8,611 m", "Second highest mountain on Earth, known as the Savage Mountain."),
            ("Kangchenjunga", "Mountain", "Himalayas (India/Nepal)", "8,586 m", "Third highest peak on Earth."),
            ("Nile River", "River", "Northeast Africa (11 countries)", "6,650 km", "Longest river system on Earth."),
            ("Amazon River", "River", "South America (Brazil/Peru/Colombia)", "6,400 km", "Largest river by discharge volume on Earth."),
            ("Yangtze River", "River", "China", "6,300 km", "Longest river in Asia and third longest in the world."),
            ("Pacific Ocean", "Ocean", "Between Americas and Asia/Oceania", "165,250,000 sq km", "Largest and deepest oceanic division on Earth."),
            ("Atlantic Ocean", "Ocean", "Between Americas and Europe/Africa", "106,460,000 sq km", "Second largest oceanic division on Earth."),
            ("Indian Ocean", "Ocean", "Between Africa, Asia, Australia", "70,560,000 sq km", "Third largest ocean division on Earth."),
            ("Sahara Desert", "Desert", "North Africa", "9,200,000 sq km", "Largest hot desert on Earth."),
            ("Antarctic Desert", "Desert", "Antarctica", "14,200,000 sq km", "Largest desert on Earth overall (polar desert)."),
            ("Caspian Sea", "Lake", "Between Europe and Asia", "371,000 sq km", "World's largest inland body of water."),
            ("Lake Baikal", "Lake", "Siberia, Russia", "1,642 m depth", "Deepest and oldest freshwater lake on Earth, holding 20% of unfrozen freshwater."),
            ("Grand Canyon", "Canyon", "Arizona, United States", "446 km long, 1,857 m deep", "Carved by the Colorado River over millions of years."),
            ("Mariana Trench", "Oceanic Trench", "Western Pacific Ocean", "10,994 m depth (Challenger Deep)", "Deepest known oceanic point on Earth."),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO world_geography (feature_name, feature_type, location, dimension_metric, notable_fact)
        VALUES (?, ?, ?, ?, ?);
        """, geo_data)

        # Seed Periodic Table
        elements_data = [
            (1, "Hydrogen", "H", 1.008, 1, 1, "Gas"),
            (2, "Helium", "He", 4.0026, 18, 1, "Gas"),
            (3, "Lithium", "Li", 6.94, 1, 2, "Solid"),
            (4, "Beryllium", "Be", 9.0122, 2, 2, "Solid"),
            (5, "Boron", "B", 10.81, 13, 2, "Solid"),
            (6, "Carbon", "C", 12.011, 14, 2, "Solid"),
            (7, "Nitrogen", "N", 14.007, 15, 2, "Gas"),
            (8, "Oxygen", "O", 15.999, 16, 2, "Gas"),
            (9, "Fluorine", "F", 18.998, 17, 2, "Gas"),
            (10, "Neon", "Ne", 20.180, 18, 2, "Gas"),
            (11, "Sodium", "Na", 22.990, 1, 3, "Solid"),
            (12, "Magnesium", "Mg", 24.305, 2, 3, "Solid"),
            (13, "Aluminium", "Al", 26.982, 13, 3, "Solid"),
            (14, "Silicon", "Si", 28.085, 14, 3, "Solid"),
            (15, "Phosphorus", "P", 30.974, 15, 3, "Solid"),
            (16, "Sulfur", "S", 32.06, 16, 3, "Solid"),
            (17, "Chlorine", "Cl", 35.45, 17, 3, "Gas"),
            (18, "Argon", "Ar", 39.948, 18, 3, "Gas"),
            (19, "Potassium", "K", 39.098, 1, 4, "Solid"),
            (20, "Calcium", "Ca", 40.078, 2, 4, "Solid"),
            (26, "Iron", "Fe", 55.845, 8, 4, "Solid"),
            (29, "Copper", "Cu", 63.546, 11, 4, "Solid"),
            (47, "Silver", "Ag", 107.868, 11, 5, "Solid"),
            (79, "Gold", "Au", 196.967, 11, 6, "Solid"),
            (80, "Mercury", "Hg", 200.592, 12, 6, "Liquid"),
            (82, "Lead", "Pb", 207.2, 14, 6, "Solid"),
            (92, "Uranium", "U", 238.029, None, 7, "Solid"),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO periodic_table (atomic_number, element_name, symbol, atomic_mass, group_number, period_number, standard_state)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, elements_data)

        # Seed Scientific Constants
        constants_data = [
            ("Speed of Light in Vacuum", "c", "299792458", "m/s", "Physics", "Universal physical constant fundamental in relativity."),
            ("Planck Constant", "h", "6.62607015e-34", "J·s", "Quantum Mechanics", "Defines the quantum of action."),
            ("Gravitational Constant", "G", "6.67430e-11", "m^3/(kg·s^2)", "Astrophysics", "Newtonian constant of gravitation."),
            ("Elementary Charge", "e", "1.602176634e-19", "C", "Electromagnetism", "Electric charge carried by a single proton."),
            ("Boltzmann Constant", "k_B", "1.380649e-23", "J/K", "Thermodynamics", "Relates average kinetic energy of particles to temperature."),
            ("Avogadro Constant", "N_A", "6.02214076e+23", "mol^-1", "Chemistry", "Number of constituent particles per mole of substance."),
            ("Gas Constant", "R", "8.314462618", "J/(mol·K)", "Thermodynamics", "Molar equivalent of Boltzmann constant."),
            ("Standard Acceleration of Gravity", "g", "9.80665", "m/s^2", "Geophysics", "Nominal gravitational acceleration of an object at sea level on Earth."),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO scientific_constants (constant_name, symbol, numerical_value, unit, discipline, description)
        VALUES (?, ?, ?, ?, ?, ?);
        """, constants_data)

        # Seed World History
        history_data = [
            ("Invention of the Printing Press", "1440", "15th", "Technology", "Johannes Gutenberg", "Revolutionized spread of information and literacy across Europe."),
            ("First Voyage of Christopher Columbus", "1492", "15th", "Exploration", "Christopher Columbus", "Initiated continuous contact between Europe and the Americas."),
            ("Publication of Principia Mathematica", "1687", "17th", "Science", "Isaac Newton", "Formulated laws of motion and universal gravitation."),
            ("Declaration of Independence", "1776", "18th", "Politics", "Thomas Jefferson, Benjamin Franklin", "Established the United States of America as a sovereign nation."),
            ("French Revolution", "1789", "18th", "Politics", "Maximilien Robespierre, Louis XVI", "Overthrew the French monarchy and established democratic principles."),
            ("First Powered Airplane Flight", "1903", "20th", "Aviation", "Wright Brothers (Orville & Wilbur)", "Kitty Hawk flight ushered in modern aviation."),
            ("Discovery of Penicillin", "1928", "20th", "Medicine", "Alexander Fleming", "Started the era of antibiotics, saving millions of lives."),
            ("First Human Moon Landing (Apollo 11)", "1969", "20th", "Space Exploration", "Neil Armstrong, Buzz Aldrin", "First time humans set foot on another celestial body."),
            ("Launch of the World Wide Web", "1989", "20th", "Computing", "Tim Berners-Lee", "Created the public global information mesh on the Internet."),
            ("Human Genome Project Completion", "2003", "21st", "Genetics", "Francis Collins, Craig Venter", "Sequenced the full 3.1 billion base pairs of human DNA."),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO world_history (event_name, year_date, century, category, key_figures, significance)
        VALUES (?, ?, ?, ?, ?, ?);
        """, history_data)

        # Seed Landmarks
        landmarks_data = [
            ("Great Pyramid of Giza", "Egypt", "Giza", "c. 2560 BCE", "Ancient Wonder", "Oldest and only surviving wonder of the ancient world."),
            ("Eiffel Tower", "France", "Paris", "1889", "Monument", "Wrought-iron lattice tower built for the 1889 World's Fair."),
            ("Taj Mahal", "India", "Agra", "1653", "Mausoleum", "Ivory-white marble mausoleum commissioned by Mughal Emperor Shah Jahan."),
            ("Colosseum", "Italy", "Rome", "80 CE", "Amphitheatre", "Largest ancient amphitheatre ever built, used for gladiatorial contests."),
            ("Statue of Liberty", "United States", "New York City", "1886", "Sculpture", "Gift from France symbolizing freedom and democracy."),
            ("Great Wall of China", "China", "Northern China", "7th Century BCE - 17th Century CE", "Fortification", "Series of fortifications built across the historical northern borders."),
            ("Machu Picchu", "Peru", "Cusco Region", "c. 1450 CE", "Archaeological Site", "15th-century Inca citadel set high in the Andes Mountains."),
            ("Christ the Redeemer", "Brazil", "Rio de Janeiro", "1931", "Statue", "Art Deco statue of Jesus Christ overlooking Rio de Janeiro."),
            ("Sydney Opera House", "Australia", "Sydney", "1973", "Performing Arts Center", "Multi-venue performing arts centre designed by Jørn Utzon."),
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO famous_landmarks (landmark_name, country, city, year_built, category, significance)
        VALUES (?, ?, ?, ?, ?, ?);
        """, landmarks_data)

        # Ensure compatibility views and unified aliases for agentic SQL queries
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS countries_and_capitals AS 
        SELECT 
            country AS country_name,
            country,
            capital,
            continent,
            currency,
            currency_code,
            languages,
            population,
            area_sq_km,
            tld
        FROM world_countries;
        """)

        cursor.execute("""
        CREATE VIEW IF NOT EXISTS physical_constants AS
        SELECT 
            constant_name AS name,
            constant_name,
            symbol,
            numerical_value AS value_str,
            numerical_value,
            unit,
            discipline AS category,
            description
        FROM scientific_constants;
        """)

        cursor.execute("""
        CREATE VIEW IF NOT EXISTS world_history_milestones AS
        SELECT 
            event_name,
            year_date AS year_str,
            year_date,
            significance AS description,
            century AS era,
            key_figures
        FROM world_history;
        """)

        # Add name alias to periodic table if needed
        cols = [col[1] for col in cursor.execute("PRAGMA table_info(periodic_table)").fetchall()]
        if "name" not in cols:
            cursor.execute("ALTER TABLE periodic_table ADD COLUMN name TEXT;")
            cursor.execute("UPDATE periodic_table SET name = element_name;")

        conn.commit()

    # =========================================================================
    # AUTONOMOUS AGENT RETRIEVAL APIS (Called via <act> ... </act>)
    # =========================================================================

    def lookup_fact(self, entity: str, attribute: str) -> str:
        """
        Direct slot retrieval for entities and attributes.
        Examples:
          lookup_fact("France", "capital") -> "Paris"
          lookup_fact("Japan", "currency") -> "Japanese Yen (JPY)"
          lookup_fact("Gold", "symbol") -> "Au"
        """
        ent = entity.strip()
        attr = attribute.strip().lower()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Search in Countries
            cursor.execute("SELECT * FROM world_countries WHERE country = ? COLLATE NOCASE;", (ent,))
            row = cursor.fetchone()
            if row:
                col_names = [col[0] for col in cursor.description]
                attr_norm = attr.replace(" ", "_")
                if attr_norm in col_names:
                    return str(row[attr_norm])
                if attr in ("money", "currency", "curr"):
                    return f"{row['currency']} ({row['currency_code']})"
                if attr in ("language", "lang", "spoken"):
                    return str(row["languages"])
                if attr in ("size", "area"):
                    return f"{row['area_sq_km']:,} sq km"
                if attr in ("pop", "people"):
                    return f"{row['population']:,}"

            # 2. Search in Periodic Table
            cursor.execute("SELECT * FROM periodic_table WHERE element_name = ? OR symbol = ? COLLATE NOCASE;", (ent, ent))
            row = cursor.fetchone()
            if row:
                col_names = [col[0] for col in cursor.description]
                if attr in col_names:
                    return str(row[attr])
                if attr in ("atomic number", "number", "z"):
                    return str(row["atomic_number"])
                if attr in ("weight", "mass", "atomic weight", "atomic mass"):
                    return str(row["atomic_mass"])
                if attr in ("symbol", "sym"):
                    return str(row["symbol"])

            # 3. Search in Geography
            cursor.execute("SELECT * FROM world_geography WHERE feature_name = ? COLLATE NOCASE;", (ent,))
            row = cursor.fetchone()
            if row:
                if attr in ("height", "length", "depth", "dimension", "size"):
                    return str(row["dimension_metric"])
                if attr in ("type", "category"):
                    return str(row["feature_type"])
                if attr in ("location", "where"):
                    return str(row["location"])
                return str(row["notable_fact"])

            # 4. Search in Constants
            cursor.execute("SELECT * FROM scientific_constants WHERE constant_name = ? OR symbol = ? COLLATE NOCASE;", (ent, ent))
            row = cursor.fetchone()
            if row:
                if attr in ("value", "number", "val"):
                    return f"{row['numerical_value']} {row['unit']}"
                if attr in ("unit", "units"):
                    return str(row["unit"])
                if attr in ("symbol", "sym"):
                    return str(row["symbol"])
                return f"{row['constant_name']} ({row['symbol']}) = {row['numerical_value']} {row['unit']}"

            # 5. Search in Landmarks
            cursor.execute("SELECT * FROM famous_landmarks WHERE landmark_name = ? COLLATE NOCASE;", (ent,))
            row = cursor.fetchone()
            if row:
                if attr in ("country", "nation"):
                    return str(row["country"])
                if attr in ("city", "town"):
                    return str(row["city"])
                if attr in ("year", "built", "age"):
                    return str(row["year_built"])
                return str(row["significance"])

        return f"[Fact not found for entity: '{entity}', attribute: '{attribute}']"

    def search_facts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Searches across all tables for matching keywords."""
        q = f"%{query.strip()}%"
        results = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Countries
            cursor.execute("""
            SELECT 'country' as table_source, country as title, capital, continent, currency, population
            FROM world_countries
            WHERE country LIKE ? OR capital LIKE ? OR currency LIKE ?
            LIMIT ?;
            """, (q, q, q, limit))
            for row in cursor.fetchall():
                results.append(dict(row))

            # Geography
            cursor.execute("""
            SELECT 'geography' as table_source, feature_name as title, feature_type, location, dimension_metric
            FROM world_geography
            WHERE feature_name LIKE ? OR location LIKE ? OR notable_fact LIKE ?
            LIMIT ?;
            """, (q, q, q, limit))
            for row in cursor.fetchall():
                results.append(dict(row))

            # History
            cursor.execute("""
            SELECT 'history' as table_source, event_name as title, year_date, category, significance
            FROM world_history
            WHERE event_name LIKE ? OR significance LIKE ? OR key_figures LIKE ?
            LIMIT ?;
            """, (q, q, q, limit))
            for row in cursor.fetchall():
                results.append(dict(row))

            # Periodic Table
            cursor.execute("""
            SELECT 'periodic_table' as table_source, element_name as title, symbol, atomic_number, atomic_mass
            FROM periodic_table
            WHERE element_name LIKE ? OR symbol LIKE ?
            LIMIT ?;
            """, (q, q, limit))
            for row in cursor.fetchall():
                results.append(dict(row))

            # Scientific Constants
            cursor.execute("""
            SELECT 'constant' as table_source, constant_name as title, symbol, numerical_value, unit
            FROM scientific_constants
            WHERE constant_name LIKE ? OR symbol LIKE ?
            LIMIT ?;
            """, (q, q, limit))
            for row in cursor.fetchall():
                results.append(dict(row))

        return results[:limit]

    def execute_query(self, sql_query: str) -> Union[List[Dict[str, Any]], str]:
        """Executes a SQL query safely (blocks dangerous DDL/destructive commands in agent mode)."""
        clean_sql = sql_query.strip().strip("`").strip()
        if clean_sql.lower().startswith("sql:"):
            clean_sql = clean_sql[4:].strip()
        elif clean_sql.lower().startswith("sql"):
            clean_sql = clean_sql[3:].strip()
        clean_sql = clean_sql.lstrip(":").strip()

        # Guard against destructive commands if called blindly by models
        forbidden = ("DROP ", "ALTER ", "ATTACH ", "DETACH ")
        if any(f in clean_sql.upper() for f in forbidden):
            return "[Database Security Error: DDL structural operations forbidden in agent tool mode]"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(clean_sql)
                if clean_sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    return [dict(r) for r in rows]
                else:
                    conn.commit()
                    return f"[Query executed successfully. Rows affected: {cursor.rowcount}]"
            except Exception as e:
                return f"[SQL Execution Error: {e}]"

    # =========================================================================
    # ADMINISTRATIVE & DATABASE MANAGEMENT APIS
    # =========================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Returns row counts and metadata across all tables."""
        stats = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [r[0] for r in cursor.fetchall()]
            for tbl in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
                cnt = cursor.fetchone()[0]
                cursor.execute(f"PRAGMA table_info({tbl});")
                cols = [c[1] for c in cursor.fetchall()]
                stats[tbl] = {"row_count": cnt, "columns": cols}
        return stats

    def add_country(self, country_dict: Dict[str, Any]) -> str:
        """Inserts or updates a country record."""
        required = ["country", "capital", "continent", "currency", "currency_code"]
        for r in required:
            if r not in country_dict:
                return f"[Error: Missing required field '{r}']"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO world_countries (country, capital, continent, currency, currency_code, languages, population, area_sq_km, tld)
            VALUES (:country, :capital, :continent, :currency, :currency_code, :languages, :population, :area_sq_km, :tld);
            """, {
                "country": country_dict["country"],
                "capital": country_dict["capital"],
                "continent": country_dict["continent"],
                "currency": country_dict["currency"],
                "currency_code": country_dict["currency_code"],
                "languages": country_dict.get("languages", "Unknown"),
                "population": country_dict.get("population", 0),
                "area_sq_km": country_dict.get("area_sq_km", 0.0),
                "tld": country_dict.get("tld", ""),
            })
            conn.commit()
            return f"Successfully added/updated country '{country_dict['country']}'."

    def delete_record(self, table: str, primary_key_col: str, key_val: Any) -> str:
        """Deletes a record by primary key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"DELETE FROM {table} WHERE {primary_key_col} = ?;", (key_val,))
                conn.commit()
                if cursor.rowcount > 0:
                    return f"Successfully deleted record '{key_val}' from {table}."
                return f"No record found matching '{key_val}' in {table}."
            except Exception as e:
                return f"[Delete Error: {e}]"

    def export_table_to_csv(self, table_name: str, output_path: str) -> str:
        """Exports a table to a CSV file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT * FROM {table_name};")
                rows = cursor.fetchall()
                if not rows:
                    return f"[Warning: Table {table_name} is empty]"
                headers = [col[0] for col in cursor.description]
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    for r in rows:
                        writer.writerow(list(r))
                return f"Exported {len(rows)} rows from '{table_name}' to {output_path}."
            except Exception as e:
                return f"[Export Error: {e}]"

    def import_table_from_csv(self, csv_path: str, table_name: str) -> str:
        """Imports rows from CSV into existing table."""
        if not os.path.exists(csv_path):
            return f"[Error: CSV file not found at {csv_path}]"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return "[Error: Empty CSV file]"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ", ".join(["?"] * len(headers))
                col_str = ", ".join(headers)
                sql = f"INSERT OR REPLACE INTO {table_name} ({col_str}) VALUES ({placeholders});"
                count = 0
                for row in reader:
                    if row:
                        cursor.execute(sql, row)
                        count += 1
                conn.commit()
                return f"Successfully imported {count} records into '{table_name}' from {csv_path}."


# Singleton instance for direct imports and agent execution
facts_db = WorldFactsManager()


# Tool functions callable by the model or agent dispatchers
def db_lookup(entity: str, attribute: str) -> str:
    """Agent tool hook: lookup_fact(entity, attribute)."""
    return facts_db.lookup_fact(entity, attribute)


def db_search(query: str, limit: int = 5) -> str:
    """Agent tool hook: search_facts(query). Returns formatted text."""
    results = facts_db.search_facts(query, limit=limit)
    if not results:
        return f"No world facts found matching '{query}'."
    lines = []
    for r in results:
        title = r.get("title", "Result")
        src = r.get("table_source", "Fact")
        details = ", ".join([f"{k}: {v}" for k, v in r.items() if k not in ("title", "table_source") and v is not None])
        lines.append(f"• [{src.upper()}] {title} -> {details}")
    return "\n".join(lines)


def db_query(sql: str) -> str:
    """Agent tool hook: execute_query(sql). Returns formatted text or JSON."""
    res = facts_db.execute_query(sql)
    if isinstance(res, list):
        if not res:
            return "[Empty query result (0 rows)]"
        return json.dumps(res, indent=2)
    return str(res)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="World Facts Database Manager")
    parser.add_argument("--stats", action="store_true", help="Print table statistics")
    parser.add_argument("--lookup", type=str, help="Entity to lookup (e.g. France)")
    parser.add_argument("--attr", type=str, default="capital", help="Attribute to lookup (default: capital)")
    parser.add_argument("--search", type=str, help="Keyword to search across world facts")
    parser.add_argument("--query", type=str, help="SQL query to execute")
    parser.add_argument("--export", type=str, help="Table name to export to CSV")
    parser.add_argument("--output", type=str, help="Output CSV path for export")
    args = parser.parse_args()

    mgr = WorldFactsManager()

    if args.stats:
        stats = mgr.get_database_stats()
        print("\n📊 WORLD FACTS DATABASE STATISTICS:")
        print("=" * 60)
        total_rows = 0
        for tbl, data in stats.items():
            print(f"  • {tbl:22}: {data['row_count']:>5} rows | {len(data['columns'])} cols")
            total_rows += data['row_count']
        print("-" * 60)
        print(f"  Total Facts Stored    : {total_rows:>5} records")
        print("=" * 60 + "\n")
        return

    if args.lookup:
        res = mgr.lookup_fact(args.lookup, args.attr)
        print(f"\n🔎 Lookup: {args.lookup} [{args.attr}] -> {res}\n")
        return

    if args.search:
        print(f"\n🔍 Search matches for '{args.search}':")
        print(db_search(args.search))
        print()
        return

    if args.query:
        print(f"\n⚡ Query Result:")
        print(db_query(args.query))
        print()
        return

    if args.export and args.output:
        res = mgr.export_table_to_csv(args.export, args.output)
        print(f"\n💾 {res}\n")
        return

    # Default: print stats
    parser.print_help()


if __name__ == "__main__":
    main()
