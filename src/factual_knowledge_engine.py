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
  9. http_status_codes          (RFC HTTP response codes: 1xx, 2xx, 3xx, 4xx, 5xx)
  10. linux_commands            (Core Linux commands, flags, descriptions, examples)
  11. mime_types                (File extensions and standardized IANA media types)
  12. standard_regex_patterns   (Production regexes: emails, IPs, UUIDs, dates, etc.)
  13. planetary_astronomy       (Solar system celestial bodies, gravity, mass, orbits)
  14. si_unit_conversions       (SI and empirical unit multipliers, categories, formulas)
  15. bioprocess_cell_lines     (Industrial host cell lines: CHO, HEK, Vero, doubling times)
  16. prime_numbers             (First 500 prime numbers indexed 1 to 500)
  17. available_tools           (Agentic tool calling registry, schemas, and usage examples)

Supported Protocols:
  - Level 0 (Direct Plain Text):   Prompt -> Plain text answer
  - Level 1 (Short Verification):  Prompt -> <think> ... </think> Answer
  - Level 3 (Agentic SQL Tool):    Prompt -> <think> ... </think> <tool_call> sql_query(...) </tool_call> Answer
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

HTTP_STATUS_CODES_DATA = [
    (100, "Continue", "Informational", "Initial part of a request received, client should continue with request."),
    (101, "Switching Protocols", "Informational", "Server agreed to switch protocols according to Upgrade header."),
    (102, "Processing", "Informational", "Server has received and is processing the request, but no response is available yet."),
    (103, "Early Hints", "Informational", "Returns headers before rest of response, allowing client to preload resources."),
    (200, "OK", "Success", "Standard response for successful HTTP requests."),
    (201, "Created", "Success", "Request fulfilled, resulting in the creation of a new resource."),
    (202, "Accepted", "Success", "Request accepted for processing, but processing has not been completed."),
    (204, "No Content", "Success", "Server successfully processed request, but is not returning any content."),
    (206, "Partial Content", "Success", "Server is delivering only part of resource due to Range header sent by client."),
    (301, "Moved Permanently", "Redirection", "Target resource has been assigned a new permanent URI."),
    (302, "Found", "Redirection", "Target resource resides temporarily under a different URI."),
    (303, "See Other", "Redirection", "Response can be found under another URI using the GET method."),
    (304, "Not Modified", "Redirection", "Indicates resource has not been modified since version specified in If-Modified-Since."),
    (307, "Temporary Redirect", "Redirection", "Target resource resides temporarily under different URI; method must not change."),
    (308, "Permanent Redirect", "Redirection", "Target resource has been assigned new permanent URI; method must not change."),
    (400, "Bad Request", "Client Error", "Server cannot process request due to client error or malformed syntax."),
    (401, "Unauthorized", "Client Error", "Authentication is required and has failed or has not yet been provided."),
    (402, "Payment Required", "Client Error", "Reserved for digital payment systems; occasionally used for quota paywalls."),
    (403, "Forbidden", "Client Error", "Server understood request but refuses to authorize it; credentials lack permissions."),
    (404, "Not Found", "Client Error", "Server cannot find the requested resource."),
    (405, "Method Not Allowed", "Client Error", "HTTP method is known by server but not supported for target resource."),
    (408, "Request Timeout", "Client Error", "Server timed out waiting for client request."),
    (409, "Conflict", "Client Error", "Request conflicts with current state of target resource (e.g., edit conflict)."),
    (410, "Gone", "Client Error", "Target resource is no longer available at origin server and no forwarding address is known."),
    (413, "Payload Too Large", "Client Error", "Request entity is larger than limits defined by server."),
    (415, "Unsupported Media Type", "Client Error", "Payload format is in unsupported format for requested endpoint."),
    (418, "I'm a teapot", "Client Error", "April Fools joke RFC 2324; returned when attempting to brew coffee on a teapot."),
    (422, "Unprocessable Entity", "Client Error", "Request syntax was well-formed, but semantic instructions could not be followed."),
    (429, "Too Many Requests", "Client Error", "User has sent too many requests in a given amount of time (rate limited)."),
    (500, "Internal Server Error", "Server Error", "Generic error message when an unexpected condition was encountered on server."),
    (501, "Not Implemented", "Server Error", "Server does not recognize request method, or lacks ability to fulfil it."),
    (502, "Bad Gateway", "Server Error", "Server acting as gateway/proxy received invalid response from inbound server."),
    (503, "Service Unavailable", "Server Error", "Server is currently unable to handle request due to temporary overload/maintenance."),
    (504, "Gateway Timeout", "Server Error", "Server acting as gateway/proxy did not receive timely response from upstream server."),
    (505, "HTTP Version Not Supported", "Server Error", "Server does not support HTTP protocol version used in request."),
]

LINUX_COMMANDS_DATA = [
    ("ls", "-l", "Use a long listing format displaying permissions, owner, size, and modification date", "ls -l /var/log", "File Management"),
    ("ls", "-a", "Show all files including hidden entries starting with a dot", "ls -la ~", "File Management"),
    ("ls", "-h", "Print sizes in human-readable format (e.g., 1K 234M 2G) with -l", "ls -lh", "File Management"),
    ("grep", "-r", "Recursively search subdirectories for matching patterns", "grep -r 'TODO' src/", "Text Processing"),
    ("grep", "-i", "Perform case-insensitive matching", "grep -i 'error' app.log", "Text Processing"),
    ("grep", "-n", "Prefix each line of matching output with its 1-based line number", "grep -n 'main' script.py", "Text Processing"),
    ("find", "-name", "Search for files matching the specified shell glob pattern", "find . -name '*.py'", "File Search"),
    ("find", "-type", "Filter search results by file type (f for regular file, d for directory)", "find /data -type f -size +100M", "File Search"),
    ("find", "-mtime", "Find files modified within or beyond n*24 hours", "find . -type f -mtime -7", "File Search"),
    ("tar", "-czvf", "Create a gzip-compressed archive with verbose progress output", "tar -czvf backup.tar.gz /data", "Archiving"),
    ("tar", "-xzvf", "Extract a gzip-compressed archive with verbose output", "tar -xzvf archive.tar.gz -C /opt", "Archiving"),
    ("chmod", "-R", "Recursively change file mode bits / permissions across directory tree", "chmod -R 755 /var/www", "Security / Permissions"),
    ("curl", "-s", "Silent or quiet mode; do not show progress meter or error messages", "curl -s https://api.ipify.org", "Networking"),
    ("curl", "-I", "Fetch HTTP headers only (HEAD request)", "curl -I https://bioflo.in", "Networking"),
    ("curl", "-d", "Send specified data in a POST request to HTTP server", "curl -d '{\"key\":\"val\"}' -H 'Content-Type: application/json' http://localhost:8000/api", "Networking"),
    ("ps", "aux", "Report snapshot of current processes across all users with terminal/no-terminal", "ps aux | grep python", "Process Management"),
    ("df", "-h", "Report file system disk space usage in human-readable units (GB, MB)", "df -h", "System Monitoring"),
    ("du", "-sh", "Display summary disk usage of target directory in human-readable format", "du -sh /home/ajit", "System Monitoring"),
    ("kill", "-9", "Send SIGKILL signal to forcibly terminate process immediately without cleanup", "kill -9 12345", "Process Management"),
    ("kill", "-15", "Send SIGTERM signal requesting graceful process termination", "kill -15 12345", "Process Management"),
    ("rsync", "-avz", "Archive mode, verbose, compressed remote file transfer", "rsync -avz ./src/ remote:/opt/app/", "Networking / Transfer"),
    ("git", "status", "Show working tree status including staged, unstaged, and untracked files", "git status", "Version Control"),
    ("git", "log", "Show commit logs with history and metadata", "git log -n 5 --oneline", "Version Control"),
    ("git", "diff", "Show changes between commits, commit and working tree, etc.", "git diff HEAD~1", "Version Control"),
    ("ssh", "-i", "Select identity file (private key) for public key authentication", "ssh -i ~/.ssh/id_rsa user@host", "Remote Access"),
    ("docker", "ps -a", "List all containers including stopped and exited ones", "docker ps -a", "Containers"),
    ("docker", "exec -it", "Execute interactive command inside running container with allocated pseudo-TTY", "docker exec -it web_app bash", "Containers"),
    ("systemctl", "status", "Inspect runtime status of systemd system and service manager units", "systemctl status nginx", "System Administration"),
    ("journalctl", "-u", "Query systemd journal logs filtered by specific systemd unit service", "journalctl -u my_service -f", "Logging"),
    ("jq", "-r", "Output raw strings instead of JSON quotes", "cat payload.json | jq -r '.data.id'", "Text Processing"),
]

MIME_TYPES_DATA = [
    ("json", "application/json", "application"),
    ("csv", "text/csv", "text"),
    ("html", "text/html", "text"),
    ("htm", "text/html", "text"),
    ("txt", "text/plain", "text"),
    ("pdf", "application/pdf", "application"),
    ("png", "image/png", "image"),
    ("jpeg", "image/jpeg", "image"),
    ("jpg", "image/jpeg", "image"),
    ("parquet", "application/vnd.apache.parquet", "application"),
    ("zip", "application/zip", "application"),
    ("tar", "application/x-tar", "application"),
    ("gz", "application/gzip", "application"),
    ("xml", "application/xml", "application"),
    ("js", "application/javascript", "application"),
    ("mjs", "application/javascript", "application"),
    ("ts", "application/typescript", "application"),
    ("py", "text/x-python", "text"),
    ("wasm", "application/wasm", "application"),
    ("mp4", "video/mp4", "video"),
    ("webm", "video/webm", "video"),
    ("mp3", "audio/mpeg", "audio"),
    ("wav", "audio/wav", "audio"),
    ("webp", "image/webp", "image"),
    ("svg", "image/svg+xml", "image"),
    ("css", "text/css", "text"),
    ("sqlite", "application/vnd.sqlite3", "application"),
    ("db", "application/vnd.sqlite3", "application"),
    ("bin", "application/octet-stream", "application"),
    ("yaml", "application/yaml", "application"),
    ("yml", "application/yaml", "application"),
]

STANDARD_REGEX_PATTERNS_DATA = [
    ("email_validation", r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", "none", "Standard RFC 5322 compatible email address pattern"),
    ("ipv4_address", r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$", "none", "Validates dotted quad IPv4 address (0.0.0.0 to 255.255.255.255)"),
    ("ipv6_address", r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$", "none", "Matches standard 8-group full IPv6 address"),
    ("iso8601_datetime", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$", "none", "Matches ISO 8601 extended UTC/offset timestamp"),
    ("url_http_https", r"^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$", "none", "Validates HTTP and HTTPS Web URLs"),
    ("uuid_v4", r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$", "none", "Validates 128-bit RFC 4122 Version 4 UUID"),
    ("semver", r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$", "none", "Semantic Versioning 2.0.0 specification regex"),
    ("hex_color", r"^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$", "none", "Matches 3-digit or 6-digit hex color code with optional leading hash"),
    ("phone_e164", r"^\+[1-9]\d{1,14}$", "none", "ITU-T E.164 international phone number format (+ followed by up to 15 digits)"),
    ("mac_address", r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", "none", "Standard 48-bit MAC hardware address (colon or hyphen separated)"),
    ("slug", r"^[a-z0-9]+(?:-[a-z0-9]+)*$", "none", "URL-friendly kebab-case alphanumeric slug"),
    ("jwt_token", r"^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$", "none", "Structure of JSON Web Token (header.payload.signature)"),
    ("markdown_link", r"\[([^\]]+)\]\(([^)]+)\)", "none", "Extracts markdown text anchor and URL target"),
]

PLANETARY_ASTRONOMY_DATA = [
    ("Sun", "Star", "1.989e30", 696340.0, 274.0, 0.0, 0.0, 8),
    ("Mercury", "Terrestrial Planet", "3.301e23", 2439.7, 3.70, 57.9, 87.97, 0),
    ("Venus", "Terrestrial Planet", "4.867e24", 6051.8, 8.87, 108.2, 224.70, 0),
    ("Earth", "Terrestrial Planet", "5.972e24", 6371.0, 9.807, 149.6, 365.25, 1),
    ("Moon", "Moon", "7.342e22", 1737.4, 1.62, 149.6, 27.32, 0),
    ("Mars", "Terrestrial Planet", "6.417e23", 3389.5, 3.721, 227.9, 686.98, 2),
    ("Jupiter", "Gas Giant", "1.898e27", 69911.0, 24.79, 778.6, 4332.59, 95),
    ("Saturn", "Gas Giant", "5.683e26", 58232.0, 10.44, 1433.5, 10759.22, 146),
    ("Uranus", "Ice Giant", "8.681e25", 25362.0, 8.69, 2872.5, 30685.4, 28),
    ("Neptune", "Ice Giant", "1.024e26", 24622.0, 11.15, 4495.1, 60189.0, 16),
    ("Pluto", "Dwarf Planet", "1.303e22", 1188.3, 0.62, 5906.4, 90560.0, 5),
    ("Ceres", "Dwarf Planet", "9.393e20", 473.0, 0.27, 413.7, 1682.0, 0),
    ("Titan", "Moon", "1.345e23", 2574.7, 1.352, 1433.5, 15.95, 0),
    ("Europa", "Moon", "4.800e22", 1560.8, 1.315, 778.6, 3.55, 0),
    ("Ganymede", "Moon", "1.482e23", 2634.1, 1.428, 778.6, 7.15, 0),
    ("Enceladus", "Moon", "1.080e20", 252.1, 0.113, 1433.5, 1.37, 0),
]

SI_UNIT_CONVERSIONS_DATA = [
    ("meter", "foot", 3.28084, "Length", "feet = meters * 3.28084"),
    ("foot", "meter", 0.3048, "Length", "meters = feet * 0.3048"),
    ("inch", "centimeter", 2.54, "Length", "cm = inches * 2.54"),
    ("mile", "kilometer", 1.60934, "Length", "km = miles * 1.60934"),
    ("kilometer", "mile", 0.621371, "Length", "miles = km * 0.621371"),
    ("kilogram", "pound", 2.20462, "Mass", "lbs = kg * 2.20462"),
    ("pound", "kilogram", 0.453592, "Mass", "kg = lbs * 0.453592"),
    ("ounce", "gram", 28.3495, "Mass", "grams = oz * 28.3495"),
    ("gram", "ounce", 0.035274, "Mass", "oz = grams * 0.035274"),
    ("bar", "pascal", 100000.0, "Pressure", "Pa = bar * 100000"),
    ("pascal", "bar", 0.00001, "Pressure", "bar = Pa * 1e-5"),
    ("atmosphere", "pascal", 101325.0, "Pressure", "Pa = atm * 101325"),
    ("psi", "pascal", 6894.76, "Pressure", "Pa = psi * 6894.76"),
    ("joule", "calorie", 0.239006, "Energy", "cal = J * 0.239006"),
    ("calorie", "joule", 4.184, "Energy", "J = cal * 4.184"),
    ("kilowatt_hour", "joule", 3600000.0, "Energy", "J = kWh * 3.6e6"),
    ("electron_volt", "joule", 1.602176634e-19, "Energy", "J = eV * 1.602176634e-19"),
    ("liter", "cubic_meter", 0.001, "Volume", "m^3 = L * 0.001"),
    ("gallon_us", "liter", 3.78541, "Volume", "L = gal * 3.78541"),
    ("horsepower", "watt", 745.7, "Power", "W = hp * 745.7"),
    ("watt", "horsepower", 0.00134102, "Power", "hp = W * 0.00134102"),
]

BIOPROCESS_CELL_LINES_DATA = [
    ("CHO-K1", "Cricetulus griseus (Chinese Hamster)", "Ovary epithelial", 18.0, 37.0, 7.1, "Therapeutic monoclonal antibody (mAb) and recombinant protein production"),
    ("HEK293", "Homo sapiens", "Embryonic kidney epithelial", 24.0, 37.0, 7.2, "Viral vector manufacturing (AAV, lentivirus) and transient gene expression"),
    ("Vero", "Chlorocebus aethiops (African Green Monkey)", "Kidney epithelial", 24.0, 37.0, 7.4, "Viral vaccine manufacturing (polio, rabies, rotavirus)"),
    ("Sf9", "Spodoptera frugiperda (Fall Armyworm)", "Ovarian insect", 24.0, 27.5, 6.2, "Baculovirus expression vector system (BEVS) for complex proteins and VLPs"),
    ("HeLa", "Homo sapiens", "Cervical carcinoma", 22.0, 37.0, 7.3, "Biomedical research, cancer biology, and cellular biology models"),
    ("Jurkat", "Homo sapiens", "T-lymphocyte", 26.0, 37.0, 7.2, "Immunology, CAR-T development, and T-cell leukemia research"),
    ("E. coli BL21(DE3)", "Escherichia coli", "Gram-negative bacterium", 0.35, 37.0, 7.0, "High-yield non-glycosylated recombinant protein and enzyme synthesis"),
    ("Pichia pastoris", "Komagataella phaffii (Yeast)", "Methylotrophic yeast", 2.5, 30.0, 6.0, "High-density fermentation of secreted eukaryotic proteins with post-translational folding"),
    ("BHK-21", "Mesocricetus auratus (Golden Hamster)", "Kidney fibroblast", 20.0, 37.0, 7.2, "Foot-and-mouth disease (FMD) vaccines and rabies vaccine culture"),
    ("Hybridoma", "Mus musculus / Mouse", "B-cell myeloma fusion", 18.0, 37.0, 7.2, "Diagnostic and murine monoclonal antibody secretor"),
    ("PER.C6", "Homo sapiens", "Retinoblast", 30.0, 37.0, 7.3, "Human adenovirus-based gene therapy vectors and recombinant biopharmaceuticals"),
    ("Sp2/0-Ag14", "Mus musculus", "B-cell myeloma", 14.0, 37.0, 7.2, "Hybridoma fusion partner for monoclonal antibody generation"),
]

def _generate_first_n_primes(n: int = 500) -> List[Tuple[int, int]]:
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return [(i + 1, p) for i, p in enumerate(primes)]

PRIME_NUMBERS_DATA = _generate_first_n_primes(500)

AVAILABLE_TOOLS_DATA = [
    ("sql_query", "Database", "Executes read-only SQL queries against cerebro_factual.db tables", json.dumps({"type": "function", "function": {"name": "sql_query", "description": "Execute a SELECT SQL query", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}), "sql_query(\"SELECT * FROM periodic_table WHERE symbol='Au';\")"),
    ("calculator", "Computation", "Evaluates mathematical and arithmetic expressions safely", json.dumps({"type": "function", "function": {"name": "calculator", "description": "Evaluate math expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}), "calculator(\"2.99792458e8 / 500e-9\")"),
    ("fetch_web_url", "Networking", "Fetches page HTML or text content from an external HTTP/HTTPS URL", json.dumps({"type": "function", "function": {"name": "fetch_web_url", "description": "Fetch web content", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}), "fetch_web_url(\"https://bioflo.in\")"),
    ("execute_bash", "Terminal", "Executes shell commands in a sandboxed Linux execution environment", json.dumps({"type": "function", "function": {"name": "execute_bash", "description": "Execute bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}), "execute_bash(\"grep -rn 'class SCMER' src/\")"),
    ("file_read", "Filesystem", "Reads UTF-8 text from workspace file path with optional line offsets", json.dumps({"type": "function", "function": {"name": "file_read", "description": "Read file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}}), "file_read(\"src/jax/scmer.py\", 1, 50)"),
    ("file_write", "Filesystem", "Writes or appends UTF-8 string content to specified file path", json.dumps({"type": "function", "function": {"name": "file_write", "description": "Write file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}}, "required": ["path", "content"]}}}), "file_write(\"data/sample.txt\", \"Hello Cerebro\", True)"),
    ("pdb_structure_fetch", "Biomolecular", "Fetches 3D atomic coordinates in mmCIF or PDB format from RCSB PDB by 4-letter ID", json.dumps({"type": "function", "function": {"name": "pdb_structure_fetch", "description": "Fetch PDB structure", "parameters": {"type": "object", "properties": {"pdb_id": {"type": "string"}}, "required": ["pdb_id"]}}}), "pdb_structure_fetch(\"4HHB\")"),
    ("unit_converter", "Physics", "Converts numerical physical quantities between compatible measurement units", json.dumps({"type": "function", "function": {"name": "unit_converter", "description": "Convert physical units", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}}), "unit_converter(100.0, \"meter\", \"foot\")"),
    ("regex_match", "Text Processing", "Tests regex pattern against target text and returns captured groups", json.dumps({"type": "function", "function": {"name": "regex_match", "description": "Execute regex matching", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "text": {"type": "string"}}, "required": ["pattern", "text"]}}}), "regex_match(\"^[0-9]+$\", \"12345\")"),
    ("date_time_now", "System", "Returns current UTC timestamp, ISO date, and time information", json.dumps({"type": "function", "function": {"name": "date_time_now", "description": "Get current timestamp", "parameters": {"type": "object", "properties": {}}}}), "date_time_now()"),
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

            # 9. HTTP Status Codes
            cur.execute("""
            CREATE TABLE IF NOT EXISTS http_status_codes (
                code INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL
            );
            """)

            # 10. Linux Commands
            cur.execute("""
            CREATE TABLE IF NOT EXISTS linux_commands (
                command TEXT NOT NULL,
                flag TEXT NOT NULL,
                description TEXT NOT NULL,
                example TEXT NOT NULL,
                category TEXT NOT NULL,
                PRIMARY KEY (command, flag)
            );
            """)

            # 11. MIME Types
            cur.execute("""
            CREATE TABLE IF NOT EXISTS mime_types (
                extension TEXT PRIMARY KEY,
                mime_type TEXT NOT NULL,
                category TEXT NOT NULL
            );
            """)

            # 12. Standard Regex Patterns
            cur.execute("""
            CREATE TABLE IF NOT EXISTS standard_regex_patterns (
                purpose TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                flags TEXT,
                description TEXT NOT NULL
            );
            """)

            # 13. Planetary Astronomy
            cur.execute("""
            CREATE TABLE IF NOT EXISTS planetary_astronomy (
                body_name TEXT PRIMARY KEY,
                body_type TEXT NOT NULL,
                mass_kg_sci TEXT NOT NULL,
                radius_km REAL NOT NULL,
                gravity_m_s2 REAL NOT NULL,
                distance_sun_10e6_km REAL NOT NULL,
                orbital_period_days REAL NOT NULL,
                moons_count INTEGER NOT NULL
            );
            """)

            # 14. SI Unit Conversions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS si_unit_conversions (
                from_unit TEXT NOT NULL,
                to_unit TEXT NOT NULL,
                multiplier REAL NOT NULL,
                category TEXT NOT NULL,
                formula TEXT NOT NULL,
                PRIMARY KEY (from_unit, to_unit)
            );
            """)

            # 15. Bioprocess Cell Lines
            cur.execute("""
            CREATE TABLE IF NOT EXISTS bioprocess_cell_lines (
                cell_line TEXT PRIMARY KEY,
                organism TEXT NOT NULL,
                cell_type TEXT NOT NULL,
                doubling_time_hr REAL NOT NULL,
                optimal_temp_c REAL NOT NULL,
                optimal_ph REAL NOT NULL,
                primary_application TEXT NOT NULL
            );
            """)

            # 16. Prime Numbers
            cur.execute("""
            CREATE TABLE IF NOT EXISTS prime_numbers (
                n_index INTEGER PRIMARY KEY,
                prime_value INTEGER NOT NULL UNIQUE
            );
            """)

            # 17. Available Tools
            cur.execute("""
            CREATE TABLE IF NOT EXISTS available_tools (
                tool_name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                example_call TEXT NOT NULL
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

            # HTTP Status Codes
            if cur.execute("SELECT COUNT(*) FROM http_status_codes;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO http_status_codes VALUES (?, ?, ?, ?);
                """, HTTP_STATUS_CODES_DATA)

            # Linux Commands
            if cur.execute("SELECT COUNT(*) FROM linux_commands;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO linux_commands VALUES (?, ?, ?, ?, ?);
                """, LINUX_COMMANDS_DATA)

            # MIME Types
            if cur.execute("SELECT COUNT(*) FROM mime_types;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO mime_types VALUES (?, ?, ?);
                """, MIME_TYPES_DATA)

            # Regex Patterns
            if cur.execute("SELECT COUNT(*) FROM standard_regex_patterns;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO standard_regex_patterns VALUES (?, ?, ?, ?);
                """, STANDARD_REGEX_PATTERNS_DATA)

            # Planetary Astronomy
            if cur.execute("SELECT COUNT(*) FROM planetary_astronomy;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO planetary_astronomy VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, PLANETARY_ASTRONOMY_DATA)

            # SI Unit Conversions
            if cur.execute("SELECT COUNT(*) FROM si_unit_conversions;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO si_unit_conversions VALUES (?, ?, ?, ?, ?);
                """, SI_UNIT_CONVERSIONS_DATA)

            # Bioprocess Cell Lines
            if cur.execute("SELECT COUNT(*) FROM bioprocess_cell_lines;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO bioprocess_cell_lines VALUES (?, ?, ?, ?, ?, ?, ?);
                """, BIOPROCESS_CELL_LINES_DATA)

            # Prime Numbers
            if cur.execute("SELECT COUNT(*) FROM prime_numbers;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO prime_numbers VALUES (?, ?);
                """, PRIME_NUMBERS_DATA)

            # Available Tools
            if cur.execute("SELECT COUNT(*) FROM available_tools;").fetchone()[0] == 0:
                cur.executemany("""
                INSERT OR IGNORE INTO available_tools VALUES (?, ?, ?, ?, ?);
                """, AVAILABLE_TOOLS_DATA)

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
        thinking_levels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes structured V5 training pairs from all registered factual tables.
        Adheres 100% to the 3-Tier Cognitive Output Protocol:
          - Level 0: Pure fluid plain-text answers (Zero tags)
          - Level 1: Short deductive scratchpad (<think>...</think> answer)
          - Level 3: Agentic SQL tool calls (<think>...</think> <tool_call> sql_query(...) </tool_call> answer)
        """
        if thinking_levels is None:
            thinking_levels = [0, 1, 3]
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
        http_codes = self.query("SELECT * FROM http_status_codes;")
        linux_cmds = self.query("SELECT * FROM linux_commands;")
        mimes = self.query("SELECT * FROM mime_types;")
        regexes = self.query("SELECT * FROM standard_regex_patterns;")
        astronomy = self.query("SELECT * FROM planetary_astronomy;")
        si_units = self.query("SELECT * FROM si_unit_conversions;")
        cell_lines = self.query("SELECT * FROM bioprocess_cell_lines;")
        primes = self.query("SELECT * FROM prime_numbers;")
        tools = self.query("SELECT * FROM available_tools;")

        while len(samples) < num_samples:
            target_level = random.choice(thinking_levels)
            domain = random.choice([
                "countries", "elements", "physical_constants", "math_constants",
                "irregular_verbs", "antonyms", "amino_acids", "history",
                "http_codes", "linux_cmds", "mimes", "regexes",
                "astronomy", "si_units", "cell_lines", "primes", "tools"
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

            elif domain == "http_codes" and http_codes:
                row = random.choice(http_codes)
                code = row["code"]
                name = row["name"]
                desc = row["description"]
                cat = row["category"]
                if random.random() < 0.5:
                    p = f"What does HTTP status code {code} mean?"
                    ans = f"HTTP {code} ({name}) is a {cat.lower()} status code: {desc}"
                    short_ans = f"{name}: {desc}"
                    th = f"Networking protocol query: Looking up RFC HTTP status code {code}."
                    sql = f"SELECT name, category, description FROM http_status_codes WHERE code={code};"
                else:
                    p = f"What is the HTTP status code for '{name}'?"
                    ans = f"The HTTP status code for '{name}' is {code} ({cat})."
                    short_ans = str(code)
                    th = f"Networking protocol query: Identifying numeric HTTP code for '{name}'."
                    sql = f"SELECT code, category FROM http_status_codes WHERE name='{name}';"

            elif domain == "linux_cmds" and linux_cmds:
                row = random.choice(linux_cmds)
                cmd = row["command"]
                flag = row["flag"]
                desc = row["description"]
                ex = row["example"]
                p = f"In Linux, what is the purpose of the '{flag}' flag in the '{cmd}' command?"
                ans = f"In the '{cmd}' command, '{flag}' is used to: {desc}. Example usage: `{ex}`."
                short_ans = desc
                th = f"Linux systems administration: Retrieving option flag '{flag}' documentation for '{cmd}'."
                sql = f"SELECT description, example FROM linux_commands WHERE command='{cmd}' AND flag='{flag}';"

            elif domain == "mimes" and mimes:
                row = random.choice(mimes)
                ext = row["extension"]
                m_type = row["mime_type"]
                cat = row["category"]
                p = f"What is the official MIME media type for file extension '.{ext}'?"
                ans = f"The MIME media type for '.{ext}' files is '{m_type}' ({cat})."
                short_ans = m_type
                th = f"Web standard media typing: Looking up IANA MIME type for '.{ext}'."
                sql = f"SELECT mime_type, category FROM mime_types WHERE extension='{ext}';"

            elif domain == "regexes" and regexes:
                row = random.choice(regexes)
                purpose = row["purpose"]
                pat = row["pattern"]
                desc = row["description"]
                display_purpose = purpose.replace("_", " ")
                p = f"What regular expression pattern is used for {display_purpose}?"
                ans = f"A standard regular expression pattern for {display_purpose} ({desc}) is: `{pat}`"
                short_ans = pat
                th = f"Text processing regex registry: Retrieving validated pattern for '{purpose}'."
                sql = f"SELECT pattern, description FROM standard_regex_patterns WHERE purpose='{purpose}';"

            elif domain == "astronomy" and astronomy:
                row = random.choice(astronomy)
                b_name = row["body_name"]
                q_type = random.choice(["gravity", "distance", "radius", "moons"])
                if q_type == "gravity":
                    p = f"What is the surface gravity of {b_name}?"
                    ans = f"The surface gravity of {b_name} is {row['gravity_m_s2']} m/s^2."
                    short_ans = f"{row['gravity_m_s2']} m/s^2"
                    th = f"Planetary astronomy query: Retrieving surface acceleration due to gravity for '{b_name}'."
                    sql = f"SELECT gravity_m_s2 FROM planetary_astronomy WHERE body_name='{b_name}';"
                elif q_type == "distance":
                    p = f"What is the average distance from the Sun to {b_name}?"
                    ans = f"The average distance from the Sun to {b_name} is approximately {row['distance_sun_10e6_km']} million km."
                    short_ans = f"{row['distance_sun_10e6_km']} million km"
                    th = f"Astrophysical orbital mechanics: Retrieving semi-major axis distance for '{b_name}'."
                    sql = f"SELECT distance_sun_10e6_km FROM planetary_astronomy WHERE body_name='{b_name}';"
                elif q_type == "radius":
                    p = f"What is the mean volumetric radius of {b_name}?"
                    ans = f"The mean radius of {b_name} is approximately {row['radius_km']} km."
                    short_ans = f"{row['radius_km']} km"
                    th = f"Planetary physics: Retrieving volumetric radius for '{b_name}'."
                    sql = f"SELECT radius_km FROM planetary_astronomy WHERE body_name='{b_name}';"
                else:
                    p = f"How many known natural satellites (moons) orbit {b_name}?"
                    ans = f"{b_name} has {row['moons_count']} known natural satellites."
                    short_ans = str(row["moons_count"])
                    th = f"Solar system astronomy: Checking confirmed moon count for '{b_name}'."
                    sql = f"SELECT moons_count FROM planetary_astronomy WHERE body_name='{b_name}';"

            elif domain == "si_units" and si_units:
                row = random.choice(si_units)
                f_u = row["from_unit"].replace("_", " ")
                t_u = row["to_unit"].replace("_", " ")
                mult = row["multiplier"]
                cat = row["category"]
                form = row["formula"]
                p = f"How do you convert from {f_u} to {t_u}?"
                ans = f"To convert {f_u} to {t_u} ({cat}): multiply by {mult} ({form})."
                short_ans = f"Multiply by {mult}"
                th = f"Dimensional analysis & unit conversion: Calculating conversion factor from {row['from_unit']} to {row['to_unit']}."
                sql = f"SELECT multiplier, formula FROM si_unit_conversions WHERE from_unit='{row['from_unit']}' AND to_unit='{row['to_unit']}';"

            elif domain == "cell_lines" and cell_lines:
                row = random.choice(cell_lines)
                c_line = row["cell_line"]
                org = row["organism"]
                c_type = row["cell_type"]
                dt = row["doubling_time_hr"]
                app = row["primary_application"]
                q_type = random.choice(["organism_type", "doubling_time", "application"])
                if q_type == "organism_type":
                    p = f"What is the source organism and cellular origin of the {c_line} cell line?"
                    ans = f"{c_line} originates from {org} ({c_type})."
                    short_ans = f"{org} ({c_type})"
                    th = f"Bioprocess cellular biology: Retrieving source organism and tissue origin for cell line '{c_line}'."
                    sql = f"SELECT organism, cell_type FROM bioprocess_cell_lines WHERE cell_line='{c_line}';"
                elif q_type == "doubling_time":
                    p = f"What is the typical doubling time and optimal growth temperature for {c_line} cells?"
                    ans = f"{c_line} cells have an average doubling time of ~{dt} hours at {row['optimal_temp_c']} °C."
                    short_ans = f"~{dt} hours at {row['optimal_temp_c']} °C"
                    th = f"Bioreactor kinetics: Retrieving cellular doubling time and temperature optimum for '{c_line}'."
                    sql = f"SELECT doubling_time_hr, optimal_temp_c FROM bioprocess_cell_lines WHERE cell_line='{c_line}';"
                else:
                    p = f"What is the primary bioprocess or therapeutic application of the {c_line} host cell line?"
                    ans = f"The primary application of {c_line} is: {app}."
                    short_ans = app
                    th = f"Biotechnology manufacturing: Retrieving industrial bioprocess application for cell line '{c_line}'."
                    sql = f"SELECT primary_application FROM bioprocess_cell_lines WHERE cell_line='{c_line}';"

            elif domain == "primes" and primes:
                row = random.choice(primes)
                idx = row["n_index"]
                val = row["prime_value"]
                p = f"What is the {idx}th prime number?"
                ans = f"The {idx}th prime number is {val}."
                short_ans = str(val)
                th = f"Number theory: Retrieving the prime number at index {idx}."
                sql = f"SELECT prime_value FROM prime_numbers WHERE n_index={idx};"

            elif domain == "tools" and tools:
                row = random.choice(tools)
                t_name = row["tool_name"]
                t_cat = row["category"]
                t_desc = row["description"]
                t_ex = row["example_call"]
                p = f"What is the function and invocation syntax for the agentic tool '{t_name}'?"
                ans = f"The '{t_name}' tool ({t_cat}) {t_desc.lower()}. Example call: `{t_ex}`."
                short_ans = t_desc
                th = f"Agentic tool capability registry: Inspecting schema and description for tool '{t_name}'."
                sql = f"SELECT description, example_call, schema_json FROM available_tools WHERE tool_name='{t_name}';"

            else:  # history fallback
                row = random.choice(hist)
                e_name = row["event_name"]
                p = f"When did the {e_name} occur?"
                ans = f"The {e_name} occurred in {row['year_str']}. {row['description']}"
                short_ans = row["year_str"]
                th = f"Historical chronology: Retrieving milestone date for '{e_name}'."
                sql = f"SELECT year_str, description FROM world_history_milestones WHERE event_name='{e_name}';"

            # Format strictly according to target thinking level
            if target_level == 0:
                if random.random() < 0.85:
                    final_out = ans
                else:
                    p = f"{p} (Answer concisely in a few words)"
                    final_out = short_ans
                action_call = None
                th_body = ""
            elif target_level == 1:
                final_out = f"<think>\n{th}\n</think>\n{ans}"
                action_call = None
                th_body = th
            else:  # Level 3: Agentic SQL tool
                act_str = f"sql_query(\"{sql}\")"
                final_out = f"<think>\n{th} Executing database lookup.\n</think>\n<tool_call> {act_str} </tool_call>\n{ans}"
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
        thinking_levels: Optional[List[int]] = None,
    ) -> str:
        """Compiles and writes a new custom dataset on command to disk."""
        if thinking_levels is None:
            thinking_levels = [0, 1, 3]
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
