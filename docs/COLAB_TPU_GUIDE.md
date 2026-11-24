# CerebroSpheroid V5 Small: Consolidated Google Colab TPU Workspace
**Target Model:** CerebroSpheroid V5 Small (105.89M Parameters, 16 Shells, 40 Shortcut Tracts)  
**Primary Execution Engine:** Pure JAX on Cloud TPU (v2-8, v3-8, v4-8, v5e, v6e)

---

## 1. Quick Start

Open [`CerebroSpheroid_V5_Colab_TPU_JAX.ipynb`](CerebroSpheroid_V5_Colab_TPU_JAX.ipynb) directly in Google Colab:
1. Select **Runtime -> Change runtime type -> TPU** (v5e-1 / v2-8 / v3-8).
2. Run **Cells 1 to 4** to verify your TPU device and environment.
3. Run **Cell 5** to begin full-dataset pre-training from scratch ($60,000$ steps $\approx 15.7$ Billion loss-bearing tokens).
4. Run **Cell 8** for live interactive prompt generation from your saved checkpoint.

---

## 2. Inventory of Consolidated Files

| File / Folder | Purpose |
|---|---|
| [`CerebroSpheroid_V5_Colab_TPU_JAX.ipynb`](CerebroSpheroid_V5_Colab_TPU_JAX.ipynb) | Master interactive Google Colab notebook for the complete 3-stage lifecycle (Pretrain $\to$ Midtrain $\to$ SFT $\to$ Inference). |
| [`train_v5_small_colab_tpu.py`](train_v5_small_colab_tpu.py) | Standalone production CLI trainer supporting multi-device TPU `pmap`, continuous 4096-token sequence packing, and token-weighted streaming. |

---

## 3. The 3-Stage Training Pipeline

### Stage 1: Full-Dataset Contiguous Pre-training (~15.7B Tokens)
- **Objective:** Pure language modeling, world entities, grammar, and code syntax.
- **Mix:** 85% Broad Web (`HuggingFaceFW/fineweb` sample-10BT) + 15% Code (`The Stack v2`).
- **Supervision:** 100% loss-bearing contiguous tokens across unbroken 4,096-token sequences (zero prompt masking, zero prompt slicing).
```bash
python colab/train_v5_small_colab_tpu.py \
    --mode contiguous_pretrain \
    --stage pretrain \
    --total-window 4096 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 2e-3 \
    --dtype bf16 \
    --total-steps 60000 \
    --save-every 250 \
    --milestone-every 1000 \
    --probe-every 500 \
    --resume \
    --checkpoint-dir /content/checkpoints/v5_small_scratch
```

### Stage 2: Deep STEM Reasoning Mid-training (15.7B Tokens)
- **Objective:** Deep STEM reasoning, mathematical proofs, algorithmic code, pedagogical textbooks, and 4k associative retrieval memory.
- **Mix:** 40% FineWeb-Edu ($\ge 3.5$) + 30% FineMath + 15% Cosmopedia v2 + 15% Python Code.
- **Supervision:** 100% loss-bearing contiguous sequences. **Strictly ZERO SFT contamination** (no conversational wrappers, no chat templates, no prompt masking).
- **Target Ingestion:** 15.7 Billion tokens (60,000 steps at batch 64, window 4,096).
- **Cumulative Tokens:** ~31.4 Billion tokens total (Pretraining + Midtraining).
```bash
python colab/train_v5_small_colab_tpu.py \
    --mode contiguous_pretrain \
    --stage midtrain \
    --total-window 4096 \
    --lr 5e-4 \
    --total-steps 60000 \
    --checkpoint-dir /content/checkpoints/v5_small_midtrain
```

### Stage 3: Agentic SFT & Latent Deliberation (~500M Tokens)
- **Objective:** DeepSeek-R1 style `<think> ... </think>` latent deliberation formatting, direct conversational answers, and deterministic `<tool_call> ... </tool_call>` execution (`CALCULATOR`, `PYTHON`, `BASH`, `FACTUAL_DB`).
- **Mix:** High-quality reasoning traces (OpenThoughts, NuminaMath, Bespoke-Stratos), verified tool trajectories, and curated SmolTalk.
- **Supervision:** Prompt-masked ($-100$ on user prompts; loss applies strictly to deliberation, tool invocations, and answers).
- **Volume:** ~100k curated trajectories (~500 Million tokens).

### Stage 4: Long-Context Extension & Fixed-Point Annealing (~1B Tokens)
- **Objective:** 4k needle-in-a-haystack retrieval consolidation, multi-hop context reasoning, fixed-point attractor energy minimization, and RLVR verification.
- **Mix:** Synthesized long-context retrieval needles (1k -> 2k -> 4k tokens), mathematical verification trajectories (GSM8k, MATH), and python execution sandboxes.
- **Supervision:** Verification rewards (RLVR / GRPO) and learning rate cooldown to 0.0.
- **Volume:** ~1.0 Billion tokens (~4,000 steps).

---

## 4. Google Colab & Kaggle Resumption Safety
All training commands automatically save checkpoints to local fast scratch storage `/content/checkpoints/...` and sync directly to `bioflo.in`:
- `v5_small_scratch_latest.pt` (overwritten every 250 steps).
- `v5_small_scratch_best.pt` (saved whenever held-out loss improves).
- `v5_small_scratch_step_{step}.pt` (permanent milestone archives every 1,000 steps).

If Colab disconnects or reaches maximum session limits, simply re-run with `--resume` to resume from the exact step and optimizer state.
