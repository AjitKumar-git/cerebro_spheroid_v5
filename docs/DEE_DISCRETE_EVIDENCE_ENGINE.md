# DEE — Discrete Evidence Engine

DEE is an ultra-high-performance, deterministic, non-neural operating system for **data integrity investigation, schema verification, epistemic discovery, and pretraining data curation** written in Rust.

Its core operating principle:

> Unusual data is evidence to investigate—not proof of corruption. Factual truth is mathematically grounded in relational manifolds, graph balance, and algorithmic description length.

---

## 1. Engine Architecture & Subsystems

```
                                    +-----------------------------------------+
                                    |        DEE CLI & Ingestion Layer        |
                                    |        (CSV, JSONL, Raw Strings)        |
                                    +-----------------------------------------+
                                                         |
         +-----------------------------------------------+-----------------------------------------------+
         |                                               |                                               |
         v                                               v                                               v
+------------------------+                     +------------------------+                     +------------------------+
|   M0: Data Profiler    |                     | M1: Schema Contracts   |                     | M2: Evidence Ledger    |
|   - Robust Median, MAD |                     | - Inferred contracts   |                     | - Finite-state ledger  |
|   - Outlier Detection  |                     | - Categorical domains  |                     | - Discrete audit trail |
+------------------------+                     +------------------------+                     +------------------------+
         |                                               |                                               |
         +-----------------------------------------------+-----------------------------------------------+
                                                         |
         +-----------------------------------------------+-----------------------------------------------+
         |                       |                       |                       |                       |
         v                       v                       v                       v                       v
+------------------+   +-------------------+   +-------------------+   +-------------------+   +-------------------+
| M3: Text Metrics |   | M4: Epistemic     |   | M5: Factual       |   | M6: Temporal      |   | M7: Math & Code   |
| - Shannon        |   | - Markov SPRT     |   | - RotatE Manifold |   | - Allen's Interval|   | - LaTeX Delimiter |
| - Stopwords      |   | - Cheeger \Phi(S) |   | - Harary Balance  |   |   Algebra         |   | - AST Syntax      |
| - Punctuation    |   | - Echo Isolation  |   | - MDL Complexity  |   | - Anachronisms    |   | - Minified Purge  |
+------------------+   +-------------------+   +-------------------+   +-------------------+   +-------------------+
                                                         |
         +-----------------------------------------------+-----------------------------------------------+
         |                                                                                               |
         v                                                                                               v
+------------------------------------+                                          +------------------------------------+
| M8: MinHash LSH Deduplication      |                                          | M9: Pedagogical / Curriculum Tier  |
| - 128 universal hash permutations  |                                          | - Flesch-Kincaid Grade Level       |
| - Banded LSH (b=16, r=8, J >= 0.8) |                                          | - Dale-Chall 3k Readability        |
| - Disjoint-Set Union clustering    |                                          | - Academic Word List (AWL) Density |
| - Longest winner preservation      |                                          | - SmolLM2 Stages 1 to 4 Routing    |
+------------------------------------+                                          +------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |   End-to-End Multi-Threaded Pipeline    |
                                    |   - Rayon data-parallel streaming       |
                                    |   - Throughput > 10,000 docs/sec        |
                                    |   - Partitioned outputs (Stage 1-4)     |
                                    +-----------------------------------------+
```

---

## 2. Mathematical Formulations

### A. RotatE Relational Manifold Energy
Entity embeddings exist in complex normed space $\mathbf{h}, \mathbf{t} \in \mathbb{C}^d$, with relations acting as rotational phases $\mathbf{r} = e^{i \boldsymbol{\theta}_r} \in (S^1)^d$.
The relational energy of an asserted triplet is:
$$E(h, r, t) = \|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|_2$$
- In canonical science, $E(\text{Earth}, \text{Orbits}, \text{Sun}) \approx 0$.
- In geocentric or flat Earth claims, $E(\text{Sun}, \text{Orbits}, \text{Earth}) \ge 1.5$ and $E(\text{Earth}, \text{HasShape}, \text{FlatDisc}) \ge 1.7$, triggering explicit energy violations.

### B. Harary's Balance Theorem & Signed Laplacian Frustration
Assertions form a signed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \sigma)$ where $\sigma: \mathcal{E} \to \{+1, -1\}$.
- Under Harary's Structural Balance Theorem (1953), a graph is balanced if and only if every cycle has an even number of negative edges:
  $$\prod_{e \in C} \sigma(e) = +1 \quad \forall C \subset \mathcal{G}$$
- Contradictory claims create odd-negative cycles. The signed Laplacian matrix is $\bar{\mathbf{L}} = \mathbf{D} - \mathbf{A}_\sigma$.
- Under Hou's theorem (2005), the smallest eigenvalue $\lambda_1(\bar{\mathbf{L}}) = 0$ if and only if the graph is balanced. Frustrated systems yield $\lambda_1(\bar{\mathbf{L}}) > 0$.

### C. Minimum Description Length (MDL) Complexity
Occam's razor formalizes that the true model minimizes total description length:
$$L(\mathcal{D}) = L(\mathcal{M}) + L(\mathcal{D} \mid \mathcal{M})$$
- Pseudoscience patches contradictions with ad-hoc epicycles ("ice wall guards", "firmament dome", "NASA CGI").
- DEE computes excess complexity: $\Delta L = L(\mathcal{M}_{\text{conspiracy}}) - L(\mathcal{M}_{\text{canonical}})$. An excess $\Delta L \ge 7.5\text{ bits}$ rejects the passage.

### D. James Allen's Temporal Interval Algebra (1983)
Evaluates mutual temporal relations between entity lifespans $[s_A, e_A]$ and technology intervals $[s_B, e_B]$:
$$\mathcal{R}_{\text{Allen}} = \{\text{before}, \text{meets}, \text{overlaps}, \text{during}, \text{starts}, \text{finishes}, \text{equals}, \dots\}$$
- An interaction asserted where $e_A < s_B$ (e.g. *Napoleon using an iPhone*) triggers an immediate `TemporalAnachronismContradiction` violation.

### E. MinHash Locality-Sensitive Hashing (LSH)
- 128 universal hash permutations partitioned into $b = 16$ bands of $r = 8$ rows.
- Probability of candidate match: $P = 1 - (1 - s^r)^b$.
- For Jaccard similarity $s \ge 0.80$, $P \ge 0.998$. Exact shingle set intersection confirms pairs before Disjoint-Set Union (DSU) clustering.

---

## 3. CLI Subcommands & Capabilities

### 1. Full Multi-Threaded Staged Pipeline (`pipeline`)
Streams a dataset through parallel Rayon threads, runs LSH deduplication, quality filtering, and partitions clean documents directly into SmolLM2 curriculum stages:
```bash
./tools/dee/target/release/dee pipeline input_dataset.csv -o output_dir/
```
Output partitions:
- `stage1_fundamental.csv` (FKGL $\le 8.5$, $\rho_{\text{AWL}} < 0.04$)
- `stage2_intermediate.csv` ($8.5 < \text{FKGL} \le 12.0$)
- `stage3_advanced.csv` ($12.0 < \text{FKGL} \le 16.0$)
- `stage4_research.csv` ($\text{FKGL} > 16.0 \lor \rho_{\text{AWL}} \ge 0.12$)

### 2. MinHash LSH Deduplication (`dedup`)
```bash
./tools/dee/target/release/dee dedup input.csv -o deduplicated.csv --threshold 0.80
```

### 3. Pedagogical Readability Grading (`grade`)
```bash
./tools/dee/target/release/dee grade "The empirical methodology incorporates longitudinal cohort regression analyses."
```

### 4. Deterministic Factual Verification (`verify-facts`)
```bash
./tools/dee/target/release/dee verify-facts "The Sun revolves around the Earth while the Earth remains motionless."
```

### 5. Temporal Interval Anachronism Verification (`verify-temporal`)
```bash
./tools/dee/target/release/dee verify-temporal "During the Battle of Waterloo, Napoleon used an iPhone."
```

### 6. Mathematical & Formula Integrity (`verify-math`)
```bash
./tools/dee/target/release/dee verify-math "Formula $E = mc^2$ without errors"
```

### 7. Code Syntax & AST Parsing (`verify-code`)
```bash
./tools/dee/target/release/dee verify-code "def add(x, y):\n    return x + y"
```

### 8. Geopolitical Perspective Attribution (`verify-geopolitics`)
```bash
./tools/dee/target/release/dee verify-geopolitics "Under international law, Beijing maintains the One China principle regarding Taiwan, while Taipei exercises independent democratic governance."
```

### 9. Toxicity & Dehumanization Audit (`audit-toxicity`)
```bash
./tools/dee/target/release/dee audit-toxicity "These immigrants are an infestation of parasites that must be exterminated like vermin."
```

### 10. Conservation Law & Pseudoscience Filter (`verify-pseudoscience`)
```bash
./tools/dee/target/release/dee verify-pseudoscience "Our breakthrough over-unity perpetual motion machine produces infinite clean free energy without fuel."
```

### 11. Systematic Quality Filter (`filter`)
```bash
./tools/dee/target/release/dee filter input.csv -o clean.csv --min-words 30 --min-grammar 0.50
```

---

## 4. Test Suite

Run all unit and mathematical property tests:
```bash
cargo test --manifest-path tools/dee/Cargo.toml
```

All **43 test suites** verify:
- RotatE complex manifold rotation invariants & Harary signed frustration
- MDL excess complexity quantification & Cheeger graph conductance
- Perspective attribution operator \Pi(T) & geopolitical dogma detection
- Dehumanization projection matrices & biological pestilence vector tracking
- Thermodynamic (\Delta S < 0) and Avogadro limit violation checking
- Scientific and medical debunking retention (zero false positives on refutations)
- James Allen temporal interval contradiction logic
- MinHash LSH near-duplicate clustering & Jaccard thresholding
- Flesch-Kincaid & Dale-Chall reading level classification
- LaTeX formula delimiter balance & OCR symbol filtering
- Code bracket balance, comment ratios, and minified asset rejection
- Schema contract validation and discrete evidence ledgers
