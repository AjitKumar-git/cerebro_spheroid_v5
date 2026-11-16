import json

def patch_sft_notebook(path: str = "colab/CerebroSpheroid_V5_Colab_SFT.ipynb"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 0: Title and Overview
    nb["cells"][0]["source"] = [
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AjitKumar-git/cerebro_spheroid_v5/blob/main/colab/CerebroSpheroid_V5_Colab_SFT.ipynb)\n",
        "\n",
        "# 🧠 CerebroSpheroid V5: Cognitive Reasoning & Agentic SFT\n",
        "### 3-Tier Cognitive Emission (<think> Deliberation) • Grounded SQLite Facts • Native JAX MXU • Google Colab TPU v5e / v6e / GPU\n",
        "\n",
        "This notebook fine-tunes **CerebroSpheroid V5** starting from your pre-trained Step 900k foundation model (`v5_small_latest_step900k.pt` / `tpu_v5e_1_latest.pt`)\n",
        "into an explicit reasoning and agentic model following the **3-Tier Cognitive Architecture**:\n",
        "- **💭 Deliberation (`<think> ... </think>`)**: Subconscious multi-cycle scratchpad for hypothesis formulation, logical deduction, step-by-step math derivation, and self-reflection.\n",
        "- **⚡ Non-Conversational Action (`<act> ... </act>`)**: Structured execution boundary for external tools (AST calculator, SQLite database SQL queries, Python scripts, system shell, Mermaid diagrams).\n",
        "- **🗣️ Conversational Speech & Answering (Plain Text to EOS)**: Fluid, natural human dialogue grounded directly in tool observations.\n",
        "\n",
        "---\n",
        "### 📦 Grounded Cognitive SFT Dataset (`dataset_source=\"chat_thinking\"`):\n",
        "100% offline, deterministic, and self-contained (15,500 samples in `data/chat_thinking_sft_15k.jsonl`, zero HuggingFace rate limits):\n",
        "1. **40% Conversational Dialogue & Empathy (6,000 samples)**: Friendly greetings, self-identity, empathy, and intuitive layman definitions (`Gate: [1, 0, 1]`).\n",
        "2. **20% Database SQL Fact Retrieval (3,000 samples)**: Grounded relational queries across sovereign nations, terrain geography, historical milestones, chemical elements, and physical constants (`Gate: [1, 1, 1]`).\n",
        "3. **15% Calculator & Arithmetic (2,250 samples)**: Sandboxed AST-evaluated mathematical reasoning preventing arithmetic drift and syntax hallucination (`Gate: [1, 1, 1]`).\n",
        "4. **15% Python Code Synthesis (2,250 samples)**: Algorithmic problem-solving, data structures, and typed functions (`Gate: [1, 1, 1]`).\n",
        "5. **10% Agentic Modalities & Self-Correction (2,000 samples)**: Structured JSON tools, regex, shell, Mermaid diagrams, plus 500 error-recovery traces for resilient self-correction (`Gate: [1, 1, 1]`).\n",
        "\n",
        "---\n",
        "### ⚡ Architectural & Training Enhancements:\n",
        "- **Expanded Context Budget**: `--prompt_len 512` and `--seq_len 1024` (1,536 total tokens) prevents lengthy `<think>` scratchpads from truncating actions or answers.\n",
        "- **Auxiliary Temporal Router Supervision**: Trained via Binary Cross-Entropy (0.15 * BCE) so the model dynamically learns to route between Thinking, Action, and Speech from manifold prompt representations.\n",
        "- **Sensory Afference ReAct Loop**: Tool calls emitted in `<act>` are intercepted, executed against sandboxed tools, and returned as sensory observations to ground the final answer.\n",
        "- **Intelligent Auto-Resumption**: Automatically scans Google Drive, local disk, and `bioflo.in` for existing SFT checkpoints (`{preset}_sft_latest.pt` / `_best.pt`) and resumes seamlessly from the exact step.\n",
        "- **Google Drive Persistence**: Auto-mounts Google Drive and mirrors checkpoints to `/content/drive/MyDrive/cerebro_v5_checkpoints` so progress is preserved across Colab disconnects.\n",
        "- **Native TPU MXU Acceleration**: Optimized for TPU v5e / v6e (Trillium) with native BF16 matrix multiplication (150k–390k tok/s).\n",
    ]

    # Cell 1: Step 1 Markdown
    nb["cells"][1]["source"] = [
        "## 🚀 Step 1: Verify TPU Hardware & Mount Google Drive for Persistence\n",
        "Verifies connected Google Colab accelerator (TPU v5e/v6e) and mounts Google Drive for persistent checkpoint saving across session disconnects.\n",
    ]

    # Cell 3: Step 2 Markdown
    nb["cells"][3]["source"] = [
        "## 📦 Step 2: Install Dependencies & Synchronize Repository\n",
        "Pulls the latest codebase from `AjitKumar-git/cerebro_spheroid_v5`.\n",
    ]

    # Cell 5: Step 3 Markdown
    nb["cells"][5]["source"] = [
        "## 🌐 Step 3: Check Pre-Trained Foundation on Bioflo.in\n",
        "Inspects the remote cloud manifest to verify Step 900k foundation model weights (`v5_small_latest_step900k.pt` / `tpu_v5e_1_latest.pt`) are available for warm-start initialization.\n",
    ]

    # Cell 7: Step 4 Markdown
    nb["cells"][7]["source"] = [
        "## 🧬 Step 4: Launch Conversational & Reasoning SFT (Steps 0 → 20,000)\n",
        "- **Automatic Resumption**: Automatically detects previous SFT checkpoints (`{preset}_sft_latest.pt` / `_best.pt`) on Google Drive, local disk, or `bioflo.in` and resumes seamlessly.\n",
        "- **Warm-Start Baseline**: Initializes from the Step 900k foundation model (`v5_small_latest_step900k.pt` / `tpu_v5e_1_latest.pt`).\n",
        "- **Grounded Dataset (`chat_thinking`)**: 100% offline, 15,500 samples (`data/chat_thinking_sft_15k.jsonl`) with conversational empathy, SQLite facts, calculator evaluations, Python code, and self-correction traces.\n",
        "- **Expanded Context Budget**: Sensory stimulus length (`PROMPT_LEN = 512`), response generation (`SEQ_LEN = 1024`), total window 1,536 tokens.\n",
        "- **Temporal Gating Supervision**: Auxiliary BCE loss ensures the model learns when to gate `<think>` and `<act>`.\n",
        "- **Persistent Storage**: Mirrored directly to Google Drive (`/content/drive/MyDrive/cerebro_v5_checkpoints`) and synced live to `bioflo.in`.\n",
    ]

    # Cell 8: Update builder in Step 4 code
    cell8_src = "".join(nb["cells"][8]["source"])
    cell8_src = cell8_src.replace(
        "from scripts.build_layman_sft_dataset import build_layman_dataset\n    build_layman_dataset()",
        "from scripts.build_chat_thinking_dataset import build_chat_thinking_dataset\n    build_chat_thinking_dataset()",
    )
    cell8_src = cell8_src.replace(
        'print("Layman dataset builder notice: {e}")',
        'print("Chat thinking dataset builder notice: {e}")',
    )
    cell8_src = cell8_src.replace(
        "# 🧠 CerebroSpheroid V5: Layman Conversational & CoT Supervised Fine-Tuning",
        "# 🧠 CerebroSpheroid V5: Cognitive Reasoning & Agentic SFT Alignment",
    )
    cell8_src = cell8_src.replace(
        "# Stream  : Layman Conversational Dialogue + Everyday Knowledge + Python Code",
        "# Stream  : Grounded Chat + Database Facts (SQL) + Calculator + Code + Error Recovery",
    )
    cell8_src = cell8_src.replace(
        "# Delib   : Explicit <think>...</think> reasoning traces showing working to layman",
        "# Delib   : Explicit <think>...</think> scratchpad + <act>...</act> grounded tool actions",
    )
    nb["cells"][8]["source"] = [l + "\n" for l in cell8_src.split("\n")[:-1]] if cell8_src.endswith("\n") else [l + "\n" for l in cell8_src.split("\n")]

    # Cell 9: Step 4 Alternative Markdown
    nb["cells"][9]["source"] = [
        "### ⚡ Alternative: Standalone CLI Launcher\n",
        "If you prefer running via standalone terminal command instead of the programmatic cell above:\n",
    ]

    # Cell 10: Step 4 Alternative Code
    nb["cells"][10]["source"] = [
        "# ==============================================================================\n",
        '# ⚡ OPTIONAL ALTERNATIVE: Standalone CLI Launch (Commented out for "Run All")\n',
        "# ==============================================================================\n",
        "# The primary SFT runner is executed in Step 4 (Cell 8) above.\n",
        "# If you prefer running via standalone bash CLI instead of programmatic execution,\n",
        "# comment out Cell 8 and uncomment the lines below:\n",
        "#\n",
        "# !git -C /content/cerebro_v5 fetch origin main && git -C /content/cerebro_v5 reset --hard origin/main\n",
        "# !python3 /content/cerebro_v5/training/train_v5_sft.py --preset auto --prompt_len 512 --seq_len 1024 --dataset_source chat_thinking --total-steps 20000 --lr 3e-5\n",
    ]

    # Cell 11: Step 4b Markdown
    nb["cells"][11]["source"] = [
        "## 🔬 Step 4b: Launch Phase 3b: Deliberative SFT Alignment (50k Steps @ 0.5 Sampling P)\n",
        "- **Objective**: Eliminate autoregressive exposure bias and train self-correcting subconscious deliberation.\n",
        "- **Starting Checkpoint**: Automatically initializes from the Phase 3a champion (`v5_small_sft_best.pt`).\n",
        "- **Deliberation Dynamics**: $p = 0.5$ (50% internal recurrent thought feedback + 50% teacher forcing).\n",
        "- **Learning Rate**: $1.5 \\times 10^{-5} \\to 1.5 \\times 10^{-6}$ (gentle cooldown for deliberative policy).\n",
        "- **Cloud Registry**: Synchronizes to `bioflo.in` under `v5_small_sft_delib`.\n",
    ]

    # Cell 13: Step 5 Markdown
    nb["cells"][13]["source"] = [
        "## 💬 Step 5: Interactive Custom Dialogue & Agentic Tool Probe\n",
        "Test arbitrary custom questions against your fine-tuned CerebroSpheroid assistant live! Tests conversation, math calculator tool calls, database SQL queries, and error self-correction.\n",
    ]

    # Cell 14: Step 5 Code (Update prompt_len to 512, add tool execution output)
    cell14_src = "".join(nb["cells"][14]["source"])
    cell14_src = cell14_src.replace("max_length=256", "max_length=512")
    cell14_src = cell14_src.replace("ids[:256] + [0] * pad_len", "ids[:512] + [0] * pad_len")
    cell14_src = cell14_src.replace("256 - len(ids)", "512 - len(ids)")
    if 'print(f"  ↳ 🔬 Observation: {obs}")' not in cell14_src:
        old_tool_print = '    if res.get("tool_calls"):\n        print(f"  ⚡ Action: {res[\'tool_calls\']}")'
        new_tool_print = (
            '    if res.get("tool_calls"):\n'
            '        from src.tools.tool_dispatcher import execute_action\n'
            '        for act in res["tool_calls"]:\n'
            '            obs = execute_action(act)\n'
            '            print(f"  ⚡ Action: {act}")\n'
            '            if obs and not obs.startswith("[Tool Error"):\n'
            '                print(f"  ↳ 🔬 Observation: {obs}")'
        )
        cell14_src = cell14_src.replace(old_tool_print, new_tool_print)

    # Add math and database queries to test_prompts
    if '"What is the capital of France?"' not in cell14_src:
        old_prompts = '"If I have 3 apples and eat 1, how many do I have left?",'
        new_prompts = (
            '"If I have 3 apples and eat 1, how many do I have left?",\n'
            '    "What is the capital of France?",\n'
            '    "What is 48291 * 23849?",\n'
            '    "What is the atomic number and symbol for Gold?",'
        )
        cell14_src = cell14_src.replace(old_prompts, new_prompts)

    nb["cells"][14]["source"] = [l + "\n" for l in cell14_src.split("\n")[:-1]] if cell14_src.endswith("\n") else [l + "\n" for l in cell14_src.split("\n")]

    # Cell 15: Step 6 Markdown
    nb["cells"][15]["source"] = [
        "## 📊 Step 6: Inspect Checkpoints & Cloud Manifest\n",
        "Inspect saved checkpoints on local disk and verify live synchronization with the remote registry on `bioflo.in`.\n",
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Successfully patched markdown and code in {path}!")


if __name__ == "__main__":
    patch_sft_notebook("colab/CerebroSpheroid_V5_Colab_SFT.ipynb")
