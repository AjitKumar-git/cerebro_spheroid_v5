# CerebroSpheroid V5: Custom Cognitive Benchmark Report

- **Checkpoint Evaluated**: `v5_small_sft_master_diverse.pt`
- **Training Milestone**: Step 2,500 | Loss: 0.4586 | Eval Loss: 0.0000
- **Timestamp**: 2026-11-16 15:32:51 UTC

## 📊 Summary Scorecard

| Cognitive Pillar | Evaluated Criteria | Score | Status |
| :--- | :--- | :---: | :---: |
| **1. Conversational Chat & Empathy** | Friendly tone, identity, zero action hallucination | **100.0%** (20/20) | 🟢 PASS |
| **2. Thinking Scratchpad Depth** | Step-by-step logic and presence of `<think>` | **100.0%** (20/20) | 🟢 PASS |
| **3. Action Execution Formatting** | Valid syntax in `<act>` (Calculator & Code) | **70.0%** (14/20) | 🟡 DEVELOPING |
| **4. Database SQL Fact Retrieval** | SQL queries targeting `cerebro_factual.db` | **100.0%** (20/20) | 🟢 PASS |
| **5. Autonomous Temporal Gating** | Action suppression on chat, activation on code/math | **85.0%** (17/20) | 🟢 PASS |
| **OVERALL COMPOSITE SCORE** | **5-Pillar Balanced Mean** | **91.0%** | 🟢 CERTIFIED |

---
*Report generated automatically by `scripts/benchmark_cerebro_cognitive.py`.*
