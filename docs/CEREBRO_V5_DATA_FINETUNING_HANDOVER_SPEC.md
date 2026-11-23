# CerebroSpheroid V5: Data & Fine-Tuning Handover Specification
**Target Audience**: Inbound Alignment / Autonomous Agent / GRPO RL Team  
**Author**: Antigravity Core Architectural Team  
**Scope Boundary**: Transfers ownership of **Dataset Engineering, SFT Alignment, Tool Dispatch, and Evaluation Probes** to allow the primary team to focus exclusively on **Group Relative Policy Optimization (GRPO)** and Concentric Core Reinforcement Learning.

---

## 1. System Overview & Division of Responsibilities

```
┌────────────────────────────────────────────────────────┐
│             GRPO RL ENGINE (PRIMARY FOCUS)             │
│ • Group Relative Advantage Estimation (G >= 4 rollouts)│
│ • Clipped Surrogate Policy Gradient (PPO/GRPO ratio)   │
│ • KL Regularization against Aligned SFT Reference Policy│
│ • Multi-Tier Exploration Rewards (RND, SVD Core Rank)  │
└───────────────────────────┬────────────────────────────┘
                            │ Handover Interface:
                            │ Datasets & SFT Checkpoints
                            ▼
┌────────────────────────────────────────────────────────┐
│       DATA & SFT ALIGNMENT (HANDED OVER / COMPLETE)    │
│ • 8,743 Multi-Turn Closed-Loop Sensory-KV Records      │
│ • Complex Sentence Answer Synthesis & Varied Phrasing  │
│ • Temporal Intent Invocations & Waiting Keywords       │
│ • Pulsating & Keep-Alive Loop Multi-Turn Samples       │
│ • Aligned Base Policy: v5_small_tool_aligned_step_500   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Directory Manifest & Repository Map

All paths are relative to workspace root:  
`/home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/`

### 2.1. Datasets & Knowledge Tables
* **[data/consolidated/02_agentic_tools/agentic_tools_sensory_grounded.jsonl](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data/consolidated/02_agentic_tools/agentic_tools_sensory_grounded.jsonl)**:
  $8,743$ verified multi-turn training records across 5 primary categories:
  - **Sensory-KV Grounded Complex Answers**: $2,122$ samples featuring complex multi-clause sentence structures.
  - **Tool Invocations with Temporal Intent**: $3,622$ samples with procedural announcements and waiting keywords.
  - **Needle-in-Sensory-KV Grounding**: $1,500$ zero-prior arbitrary tokens forcing Arcuate Fasciculus cross-resonance ($Q \cdot K^T$).
  - **Pulsating & Waiting State Checks**: $999$ samples modeling the pending/completed query loop and keep-alive heartbeats.
  - **Direct Conceptual QA**: $500$ conversational questions preserving non-tool interaction boundaries.
* **[data/factual_tables/cerebro_factual.db](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data/factual_tables/cerebro_factual.db)**:
  SQLite relational knowledge database with 17 verified tables including `countries_and_capitals` (102 rows), `periodic_table` (70 elements), and `physical_constants` (15 constants).
* **[data/tokenizer/](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/data/tokenizer/)**:
  The 32,000-vocabulary LlamaTokenizer with embedded special tokens:
  - `<s>` (ID 1), `</s>` (ID 2), `<unk>` (ID 0)
  - `<think>` (ID 29), `</think>` (ID 30)
  - `<tool_call>` (ID 31), `</tool_call>` (ID 32)
  - `<tool_response>` (ID 33), `</tool_response>` (ID 34)

### 2.2. Executable Scripts & Dispatchers
* **[scripts/build_sensory_grounded_tool_dataset.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/scripts/build_sensory_grounded_tool_dataset.py)**:
  Deterministic synthetic dataset builder implementing all complex sentence templates, temporal narration, and pulsating/waiting multi-turn loops.
* **[scripts/train_tool_augmented_sft.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/scripts/train_tool_augmented_sft.py)**:
  Mixed-precision AMP (BF16/FP16) SFT trainer with cosine LR decay, linear warmup, gradient accumulation, and bare BF16 checkpoint serialization.
* **[scripts/probe_model_with_tools.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/scripts/probe_model_with_tools.py)**:
  The 15-task diagnostic evaluation probe covering multi-step arithmetic, physical/chemical constants, geography, and Python code execution.
* **[src/tools/unified_tool_dispatcher.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/src/tools/unified_tool_dispatcher.py)**:
  Unified tool execution dispatcher supporting `calc(...)`, `factual(...)`, and `python(...)` with regex argument parsing.

### 2.3. Model Checkpoints
* **Base Foundation Checkpoint**:
  [checkpoints/v5_small_latest.pt](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/v5_small_latest.pt) (Step 202,500, 102M parameters, 16 concentric shells).
* **Tool-Aligned SFT Checkpoint (GRPO Reference Policy)**:
  [checkpoints/tool_aligned/v5_small_tool_aligned_step_500.pt](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/checkpoints/tool_aligned/v5_small_tool_aligned_step_500.pt) (Step 500, loss $0.1624$, bare BF16 format, $203\text{ MB}$).

---

## 3. Cognitive Protocol & Formatting Contracts

### A. Turn 1: Temporal Intent & Tool Invocation
The assistant announces temporal intent, emits the structured `<tool_call>`, and enters a waiting state:
```text
Calling the calculator to compute (64 * 32)...
<tool_call> calc(64 * 32) </tool_call>
Waiting for tool execution to complete...
```
or
```text
Querying the knowledge database for the capital of Germany...
<tool_call> factual('capital of Germany') </tool_call>
Waiting for database response...
```

### B. Pulsating & Waiting Keep-Alive Loops
When tool execution is asynchronous or requires latency compensation, the model supports pulsation:
1. **Pending Status Check**:
   ```text
   <|im_start|>user
   What is the capital of Germany?<|im_end|>
   <|im_start|>assistant
   Calling the tool to process request...
   <tool_call> factual('capital of Germany') </tool_call>
   Waiting for tool execution to complete...<|im_end|>
   <|im_start|>system
   Tool status check: execution pending. Has the tool responded?<|im_end|>
   <|im_start|>assistant
   Querying tool status: The tool has not responded yet. Waiting for tool execution to complete. Pulsating to maintain active state.
   ```
2. **Completed Status Check**:
   ```text
   <|im_start|>system
   Tool status check: response ready. Observation delivered.<|im_end|>
   <|im_start|>assistant
   <tool_response> Berlin </tool_response>
   Answer:
   Checking if the tool responded: Yes, the tool has responded! After querying the knowledge database, the capital of Germany is Berlin.
   ```

### C. Turn 2: Rich Complex Sentence Grounded Extraction
When `<tool_response>` is present in sensory KV, the model derives the answer using rich, natural phrasing:
* *"By calling the calculator, I got the answer 5467."*
* *"The answer is 8. By executing the prime counter algorithm in Python, we found exactly 8 primes strictly below 20."*
* *"After querying the knowledge database, the capital of Germany is Berlin."*
* *"According to the database lookup, the atomic number of Gold is 79."*
* *"The tool has responded. Calling the calculator yielded 493, so (17 * 24) + 85 equals 493."*

---

## 4. Key Engineering Discoveries & Bug Resolutions

1. **Purged Negative Reinforcement in Legacy Dataset**:
   The previous `closed_loop_sft_pairs.jsonl` paired failed lookups (`zero matches`) with hallucinated targets (`The atomic mass of Uranium is 238.03 u.`), mathematically penalizing sensory-KV attention. All failed lookups and truncated strings were purged.
2. **SentencePiece Space Token Alignment**:
   SentencePiece tokenizes `"...Answer: "` with a trailing space token (`29871`). If the target output also began with a space (`" 493"`), an artificial double space (`[29871, 29871]`) corrupted the generation boundary. Normalizing the prompt boundary to `\nAnswer:` (ending on colon token `29901`) allows `" {val}"` to match natural contiguous tokenization with 100% boundary fidelity.
3. **Syntax Standardization**:
   Purged non-standard nested JSON function-calling schemas (`{"name": ...}`). Standardized on concise functional calls (`calc(...)`, `factual(...)`, `python(...)`) that execute directly through `UnifiedToolDispatcher`.

---

## 5. Transition to GRPO (Group Relative Policy Optimization)

With dataset engineering and SFT alignment complete, the model checkpoint `v5_small_tool_aligned_step_500.pt` serves as the reference policy ($\pi_{\text{ref}}$) for GRPO reinforcement learning.

### GRPO Architecture Interface:
* **Script**: [scripts/train_cerebro_grpo_exploration.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/scripts/train_cerebro_grpo_exploration.py)
* **Reward Engine**: [src/cerebro_reward_model.py](file:///home/ajit/Documents/Startup/Active_Development/cerebro_spheroid_v5/src/cerebro_reward_model.py) (`CerebroExplorationRewardModel`)
* **Core Invariants**:
  1. Batched Group Rollouts ($G \ge 4$ candidate trajectories per prompt $q$).
  2. Outcome Grounding & Process Exploration Rewards (RND Curiosity + Concentric Core SVD Rank).
  3. Group Relative Advantage Standardization ($A_i = (R_i - \mu_R) / (\sigma_R + \epsilon)$).
  4. Clipped Surrogate Policy Gradient with KL divergence penalty against `v5_small_tool_aligned_step_500.pt`.
