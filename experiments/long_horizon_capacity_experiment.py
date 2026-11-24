#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: Long-Horizon Capacity Unlocking Experiment
Compares Arm A (Static Baseline) vs Arm C (Periodic Mid /8 + Staged Core Protection)
across 2,000 steps with early-checkpoint trajectory tracking, mid-shell ablation audits,
and automated crossover detection.
"""

import os
import sys

# Default to CPU platform to guarantee zero CUDA OOM on host GPU (4GB VRAM limit)
# Host CPU has 14GB system RAM + 14GB swap, perfectly suited for unrolled graph execution
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import math
import time
import json
import hashlib
import argparse
import subprocess
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import jax
import jax.numpy as jnp
from jax import lax

from src.spheroid_v5 import infer_v5_config_from_state_dict
from src.jax.v5_model import (
    ModelConfigV5,
    deliberate_step,
    deliberate_cycles_jax,
    encode_sensory,
    emit_step,
)
from src.jax.checkpoint import load_torch_pt_to_jax
from experiments.multitimescale_sandbox import compute_verified_causal_nll

# ==============================================================================
# 1. DATASETS: TRAINING CORPUS & STRICTLY HELD-OUT VALIDATION BENCHMARK
# ==============================================================================

TRAINING_CORPUS = [
    # Code (2)
    {"id": "train_code_quicksort", "domain": "Code", "text": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n"},
    {"id": "train_code_linkedlist", "domain": "Code", "text": "class Node:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\ndef reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev\n"},
    # Biochem (2)
    {"id": "train_biochem_glycolysis", "domain": "Biochemistry", "text": "Phosphofructokinase-1 catalyzes the irreversible phosphorylation of fructose-6-phosphate to fructose-1,6-bisphosphate, serving as the primary allosteric flux-controlling valve in glycolysis."},
    {"id": "train_biochem_polymerase", "domain": "Biochemistry", "text": "High-fidelity DNA polymerases utilize intrinsic 3'-to-5' exonucleolytic proofreading subunits to excise misincorporated mismatched nucleotides, sustaining genomic replication accuracy."},
    # Physics (2)
    {"id": "train_physics_stress_tensor", "domain": "Physics", "text": "The Maxwell stress tensor represents electromagnetic mechanical force densities and Poynting flux conservation in continuous electrodynamic field interactions."},
    {"id": "train_physics_lorentz", "domain": "Physics", "text": "Lorentz transformation matrices preserve the invariant Minkowski spacetime metric interval across inertial relativistic reference frames."},
    # Web / Distributed (2)
    {"id": "train_web_chubby", "domain": "Distributed", "text": "The Chubby distributed lock architecture utilizes coarse-grained asynchronous primary leases and Paxos consensus for cluster configuration management."},
    {"id": "train_web_consistent_hash", "domain": "Distributed", "text": "Consistent hashing arranges storage partition buckets along a circular ring topology using virtual node replications to minimize cache redistributions during node churn."},
]

HELD_OUT_VALIDATION = [
    # 1. Python Systems Code (4)
    {"id": "val_code_async_loop", "domain": "Code", "title": "Async Event Loop Scheduler", "text": "class EventLoop:\n    def __init__(self):\n        self.ready_queue = []\n    def call_soon(self, callback, *args):\n        self.ready_queue.append((callback, args))\n    def run_until_complete(self):\n        while self.ready_queue:\n            cb, args = self.ready_queue.pop(0)\n            cb(*args)\n"},
    {"id": "val_code_btree", "domain": "Code", "title": "B-Tree Node Search", "text": "def btree_search(node, key):\n    i = 0\n    while i < len(node.keys) and key > node.keys[i]:\n        i += 1\n    if i < len(node.keys) and key == node.keys[i]:\n        return (node, i)\n    elif node.is_leaf:\n        return None\n    else:\n        return btree_search(node.children[i], key)\n"},
    {"id": "val_code_threadpool", "domain": "Code", "title": "Threadpool Task Dispatcher", "text": "class ThreadPoolWorker:\n    def __init__(self, task_queue, lock):\n        self.tasks = task_queue\n        self.sync_lock = lock\n    def poll_and_execute(self):\n        with self.sync_lock:\n            if not self.tasks.empty():\n                fn, payload = self.tasks.get_nowait()\n                return fn(payload)\n        return None\n"},
    {"id": "val_code_ring_buffer", "domain": "Code", "title": "Circular Lock-Free Ring Buffer", "text": "class RingBuffer:\n    def __init__(self, capacity=1024):\n        self.buffer = [None] * capacity\n        self.head = 0\n        self.tail = 0\n        self.mask = capacity - 1\n    def push(self, item):\n        self.buffer[self.head & self.mask] = item\n        self.head += 1\n"},

    # 2. Molecular Biochemistry (4)
    {"id": "val_bio_crispr_cas9", "domain": "Biochemistry", "title": "CRISPR-Cas9 Endonuclease Mechanism", "text": "The Cas9 endonuclease forms a ribonucleoprotein complex with single-guide RNA to introduce targeted double-strand breaks adjacent to 5'-NGG protospacer adjacent motifs via coordinated RuvC and HNH catalytic nuclease domains."},
    {"id": "val_bio_atp_synthase", "domain": "Biochemistry", "title": "ATP Synthase Rotational Catalysis", "text": "Proton translocation across the mitochondrial inner membrane drives rotary movement of the c-ring subunit, transducing electrochemical potential into gamma-shaft rotational torque that cyclically alters catalytic beta-subunit conformations."},
    {"id": "val_bio_ribosome_translocation", "domain": "Biochemistry", "title": "Ribosomal Elongation Cycle", "text": "Elongation factor G hydrolyzes guanosine triphosphate to catalyze synchronized translocation of peptidyl-tRNA and messenger RNA from ribosomal A to P sites, shifting uncharged deacylated tRNA to the transient exit site."},
    {"id": "val_bio_ubiquitin_ligase", "domain": "Biochemistry", "title": "Ubiquitin-Proteasome Degradation Cascade", "text": "E3 ubiquitin-protein ligases recruit target substrates displaying degradation degron motifs, catalyzing polyubiquitin chain elongation via Lys48 isopeptide linkages to direct target polypeptides into the 26S proteasomal barrel."},

    # 3. Relativistic & Quantum Physics (4)
    {"id": "val_phys_dirac_equation", "domain": "Physics", "title": "Relativistic Dirac Electron Equation", "text": "The relativistic Dirac equation couples four-component bispinor wavefunctions through gamma matrices satisfying the Clifford anticommutation algebra, predicting intrinsic spin angular momentum and positive-energy positron antiparticles."},
    {"id": "val_phys_schwarzschild_metric", "domain": "Physics", "title": "Schwarzschild Black Hole Spacetime Metric", "text": "The static spherically symmetric vacuum Einstein field equations yield the Schwarzschild solution with coordinate singularity at the gravitational event horizon radius rs equals two GM over c squared."},
    {"id": "val_phys_bose_einstein", "domain": "Physics", "title": "Bose-Einstein Macro Condensation", "text": "Below a critical thermal de Broglie wavelength threshold, integer-spin bosonic quantum particles undergo macroscopic phase accumulation into the lowest single-particle ground state, establishing macroscopic coherence and zero-viscosity superfluidity."},
    {"id": "val_phys_yang_mills", "domain": "Physics", "title": "Non-Abelian Yang-Mills Gauge Theory", "text": "Non-Abelian gauge theories introduce Lie-algebra-valued gauge connections whose nonlinear self-interaction field strength tensor generates asymptotic freedom at high energies and non-perturbative color charge confinement."},

    # 4. Distributed Systems & Networking (4)
    {"id": "val_dist_raft_consensus", "domain": "Distributed", "title": "Raft State Machine Log Replication", "text": "The Raft consensus protocol elects a unique cluster leader using randomized election timeouts and maintains distributed state consistency by enforcing monotonic term progression and index match log replication rules."},
    {"id": "val_dist_paxos_synod", "domain": "Distributed", "title": "Paxos Two-Phase Synod Protocol", "text": "In multi-decree Paxos, ballot proposers issue Phase 1b prepare requests seeking quorum promises, subsequently broadcasting Phase 2a accept proposals bearing the highest-numbered previously accepted value among the quorum."},
    {"id": "val_dist_vector_clocks", "domain": "Distributed", "title": "Lamport Causality & Vector Clocks", "text": "Distributed nodes increment localized scalar logical timestamps upon discrete message transmissions and merge vector array maxima upon receive events, tracking strict partial order causal dependencies across concurrent processes."},
    {"id": "val_dist_gossip_epidemic", "domain": "Distributed", "title": "Gossip Protocol Epidemic Dissemination", "text": "Decentralized gossip algorithms broadcast cluster membership state by randomly selecting peer partner subsets every dissemination round, providing exponential rumor spreading and fault-tolerant eventual consistency guarantees."},
]

CHECKPOINT_STEPS = [0, 25, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000]

# ==============================================================================
# 2. UTILITY & METRIC HELPERS
# ==============================================================================

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_shell_param_metrics(tensors: Dict[str, jnp.ndarray]) -> Tuple[float, float, float]:
    g_core_sq = 0.0
    g_mid_sq = 0.0
    g_outer_sq = 0.0

    shell_tensors = [
        "core.intra_swiglu.w_gate",
        "core.intra_swiglu.w_val",
        "core.intra_swiglu.w_down",
        "core.intra_norms.weight",
        "core.shell_gains",
        "motor.shell_score",
    ]

    for name in shell_tensors:
        if name in tensors:
            val = tensors[name]
            g_core_sq += float(jnp.sum(val[0] ** 2))
            g_mid_sq += float(jnp.sum(val[1:12] ** 2)) / 11.0
            g_outer_sq += float(jnp.sum(val[12:] ** 2)) / 4.0

    if "core.w_radial_in" in tensors:
        rin = tensors["core.w_radial_in"]
        g_mid_sq += float(jnp.sum(rin[:11] ** 2)) / 11.0
        g_outer_sq += float(jnp.sum(rin[11:] ** 2)) / 4.0

    if "core.w_radial_out" in tensors:
        rout = tensors["core.w_radial_out"]
        g_mid_sq += float(jnp.sum(rout[:11] ** 2)) / 11.0
        g_outer_sq += float(jnp.sum(rout[11:] ** 2)) / 4.0

    return math.sqrt(g_core_sq), math.sqrt(g_mid_sq), math.sqrt(g_outer_sq)


def compute_radial_coupling_norms(params: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
    rin_norm = 0.0
    rout_norm = 0.0
    if "core.w_radial_in" in params:
        rin = params["core.w_radial_in"][:11]
        rin_norm = float(jnp.sqrt(jnp.sum(rin ** 2)))
    if "core.w_radial_out" in params:
        rout = params["core.w_radial_out"][:11]
        rout_norm = float(jnp.sqrt(jnp.sum(rout ** 2)))
    return rin_norm, rout_norm


# ==============================================================================
# 3. STEP FUNCTION BUILDER & OPTIMIZER UPDATE LOGIC
# ==============================================================================

def build_capacity_step_fn(config: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray], arm: str = "ArmA"):
    K = config.num_shells
    D = config.surface_dim

    def _delib_closure(p, st, buf):
        return deliberate_step(p, st, buf, config=config)

    ckpt_delib_step = jax.checkpoint(_delib_closure)

    def step_loss_fn(params, prompts, targets):
        B, total_L = targets.shape

        p_flux, p_kv = encode_sensory(params, prompts, config=config, pos_offset=0)
        r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
        init_state = (jnp.broadcast_to(p_flux, (B, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
        state, _ = deliberate_cycles_jax(params, init_state, buffers, config=config, is_training=True, sensory_kv=p_kv)

        prev_toks = jnp.concatenate([prompts[:, -1:], targets[:, :-1]], axis=1)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=config, pos_offset=prompts.shape[1], is_causal=True)
        causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
        causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

        cur_state = state
        cur_plan = jnp.zeros((B, D), dtype=jnp.float32)
        cur_trace = None
        total_loss = 0.0

        for t in range(total_L):
            curr_ctx_len = prompts.shape[1] + t + 1
            sub_k = causal_k[:, :, :curr_ctx_len, :]
            sub_v = causal_v[:, :, :curr_ctx_len, :]

            logits, nxt_state, cur_trace, cur_plan = emit_step(
                params,
                cur_state,
                (sub_k, sub_v),
                cur_plan,
                cur_trace,
                t,
                buffers,
                config=config,
                auditory_feedback=tf_flux[:, t, :],
                query_position=prompts.shape[1] + t,
            )

            tgt = targets[:, t]
            log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            step_loss = -jnp.mean(jnp.take_along_axis(log_p, tgt[:, None], axis=-1))
            total_loss = total_loss + step_loss

            prev_for_disp = cur_state

            if arm == "ArmA":
                # Arm A: Static Baseline (only Shell 0 evolves)
                cur_state = nxt_state
            elif arm == "ArmC":
                # Arm C: Periodic Mid Recurrence (/8)
                is_periodic = (t > 0) and (t % 8 == 0)
                if is_periodic:
                    delib = ckpt_delib_step(params, nxt_state, buffers)
                    cur_state = nxt_state.at[:, 1:12, :].set(0.85 * nxt_state[:, 1:12, :] + 0.15 * delib[:, 1:12, :])
                else:
                    cur_state = nxt_state

            disp_t = jnp.mean(jnp.linalg.norm(cur_state - prev_for_disp, axis=-1), axis=0)
            if t == 0:
                total_disp = disp_t
            else:
                total_disp = total_disp + disp_t

        return total_loss / total_L, total_disp / total_L

    return jax.jit(jax.value_and_grad(step_loss_fn, has_aux=True))


def apply_arm_parameter_update(
    params: Dict[str, jnp.ndarray],
    grads: Dict[str, jnp.ndarray],
    arm: str,
    step_idx: int,
) -> Dict[str, jnp.ndarray]:
    """Applies parameter updates according to arm-specific optimization schedules."""
    updated = {}

    if arm == "ArmA":
        # Arm A: Standard single learning rate for all trainable parameters
        lr_std = 1e-4
        for k, v in params.items():
            updated[k] = v - lr_std * grads[k]

    elif arm == "ArmC":
        # Arm C: Staged Differential Learning Rates
        # Stage 1 (0..250): Core frozen (0.0), Mid (2e-4), Outer (0.0)
        # Stage 2 (251..750): Core low-rate (2e-5), Mid (1e-4), Outer (0.0)
        # Stage 3 (751..2000): Core (5e-5), Mid (5e-5), Outer (0.0)
        if step_idx <= 250:
            eta_core = 0.0
            eta_mid = 2e-4
        elif step_idx <= 750:
            eta_core = 2e-5
            eta_mid = 1e-4
        else:
            eta_core = 5e-5
            eta_mid = 5e-5

        for k, v in params.items():
            g = grads[k]
            if k in ["core.shell_gains", "motor.shell_score", "core.intra_norms.weight"] or "intra_swiglu" in k:
                # Sliced shell tensors: 0 is core, 1:12 is mid, 12: is outer
                lr_mask = jnp.zeros_like(v)
                if v.ndim >= 1:
                    lr_mask = lr_mask.at[0].set(eta_core)
                    lr_mask = lr_mask.at[1:12].set(eta_mid)
                updated[k] = v - lr_mask * g
            elif k in ["core.w_radial_in", "core.w_radial_out"]:
                # Radial connectors: :11 connects up to shell 11 (mid), 11: outer
                lr_mask = jnp.zeros_like(v)
                lr_mask = lr_mask.at[:11].set(eta_mid)
                updated[k] = v - lr_mask * g
            elif "radial" in k:
                # Radial connector shared projections belong to mid pathway
                updated[k] = v - eta_mid * g
            else:
                # Core sensory, motor, deliberation controllers
                updated[k] = v - eta_core * g

    return updated


# ==============================================================================
# 4. EVALUATION & TRAJECTORY AUDITING ENGINE
# ==============================================================================

def evaluate_checkpoint_trajectory(
    params: Dict[str, jnp.ndarray],
    base_params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    cfg: ModelConfigV5,
    tokenizer: AutoTokenizer,
    arm: str,
    step_idx: int,
    train_loss: float,
    wall_clock_sec: float,
    disp_tuple: Tuple[float, float, float],
    last_grads: Optional[Dict[str, jnp.ndarray]] = None,
    seq_len: int = 16,
) -> Dict:
    """Evaluates held-out validation, mid-shell ablation, retention, and parameter metrics."""
    # 1. Held-Out Validation NLL (16 sequences)
    heldout_nlls = []
    mode_variant = "A" if arm == "ArmA" else "B"

    for item in HELD_OUT_VALIDATION:
        enc = tokenizer(item["text"], padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
        t_ids = jnp.asarray(enc["input_ids"], dtype=jnp.int32)
        p_ids = jnp.full((1, 1), 1, dtype=jnp.int32)

        nll, _, _, _ = compute_verified_causal_nll(
            params, p_ids, t_ids, buffers, cfg, variant=mode_variant, ablate_region=None
        )
        heldout_nlls.append(float(nll))

    mean_heldout_nll = float(np.mean(heldout_nlls))

    # 2. Mid-Shell Ablation NLL (Functional Pathway Usefulness Metric)
    # Zero out intermediate shells (1..11) during generation
    ablated_nlls = []
    for item in HELD_OUT_VALIDATION:
        enc = tokenizer(item["text"], padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
        t_ids = jnp.asarray(enc["input_ids"], dtype=jnp.int32)
        p_ids = jnp.full((1, 1), 1, dtype=jnp.int32)

        nll_abl, _, _, _ = compute_verified_causal_nll(
            params, p_ids, t_ids, buffers, cfg, variant=mode_variant, ablate_region="intermediate"
        )
        ablated_nlls.append(float(nll_abl))

    mean_ablated_nll = float(np.mean(ablated_nlls))
    mid_ablation_delta = mean_ablated_nll - mean_heldout_nll

    # 3. Foundation Retention NLL (Training corpus evaluated frozen)
    retention_nlls = []
    for item in TRAINING_CORPUS:
        enc = tokenizer(item["text"], padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
        t_ids = jnp.asarray(enc["input_ids"], dtype=jnp.int32)
        p_ids = jnp.full((1, 1), 1, dtype=jnp.int32)

        nll_ret, _, _, _ = compute_verified_causal_nll(
            params, p_ids, t_ids, buffers, cfg, variant="A", ablate_region=None
        )
        retention_nlls.append(float(nll_ret))
    mean_retention_nll = float(np.mean(retention_nlls))

    # 4. Parameter Displacements & Gradient Norms
    d_theta = {k: params[k] - base_params[k] for k in params}
    tc, tm, to = compute_shell_param_metrics(d_theta)
    rin_norm, rout_norm = compute_radial_coupling_norms(params)

    gc, gm, go = (0.0, 0.0, 0.0)
    if last_grads is not None:
        gc, gm, go = compute_shell_param_metrics(last_grads)

    d_core, d_mid, d_outer = disp_tuple
    gen_gap = mean_heldout_nll - train_loss

    record = {
        "step": step_idx,
        "arm": arm,
        "train_loss": train_loss,
        "heldout_nll": mean_heldout_nll,
        "gen_gap": gen_gap,
        "retention_nll": mean_retention_nll,
        "mid_ablation_delta_nll": mid_ablation_delta,
        "d_core": d_core,
        "d_mid": d_mid,
        "d_outer": d_outer,
        "delta_theta_core": tc,
        "delta_theta_mid": tm,
        "delta_theta_outer": to,
        "grad_core": gc,
        "grad_mid": gm,
        "grad_outer": go,
        "radial_in_norm": rin_norm,
        "radial_out_norm": rout_norm,
        "wall_clock_sec": wall_clock_sec,
    }

    return record


# ==============================================================================
# 5. ORCHESTRATION & RESUME-ENABLED RUNNER
# ==============================================================================

def run_arm_capacity_training(
    arm: str,
    total_steps: int = 2000,
    seed: int = 42,
    seq_len: int = 16,
    checkpoint_path: str = "checkpoints/v5_small_latest.pt",
    out_dir: str = "experiments/long_horizon_checkpoints",
    resume: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_hash = compute_file_sha256(checkpoint_path)[:16]

    print("\n" + "=" * 95)
    print(f"🚀 LAUNCHING LONG-HORIZON CAPACITY EXPERIMENT | ARM: {arm} | STEPS: {total_steps} | SEED: {seed}")
    print(f"📁 Foundation Checkpoint: {checkpoint_path} (SHA-256: {ckpt_hash})")
    print(f"📁 Output Directory: {out_dir}")
    print("=" * 95)

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cfg = infer_v5_config_from_state_dict(state, default_preset="v5_small")
    cfg.use_sensory_self_attn = True
    cfg.use_rope = False
    cfg.top_k_resonance = 0

    base_params, buffers, meta = load_torch_pt_to_jax(checkpoint_path, cfg, target_dtype=jnp.float32)

    # Deterministic batch stream per seed
    rng = np.random.RandomState(seed)
    train_texts = [item["text"] for item in TRAINING_CORPUS]
    stream_indices = [rng.choice(len(train_texts), size=4, replace=False) for _ in range(total_steps)]

    # Resume support: detect highest saved checkpoint step
    traj_path = os.path.join(out_dir, f"{arm}_trajectory.json")
    start_step = 0
    trajectory = []
    p_current = {k: jnp.array(v) for k, v in base_params.items()}

    if resume and os.path.exists(traj_path):
        try:
            with open(traj_path, "r") as f:
                trajectory = json.load(f)
            if trajectory:
                last_entry = trajectory[-1]
                saved_step = last_entry["step"]
                saved_npz = os.path.join(out_dir, f"{arm}_step_{saved_step}.npz")
                if os.path.exists(saved_npz):
                    print(f"✓ Resuming {arm} from step {saved_step} (Saved at {saved_npz})")
                    loaded = np.load(saved_npz)
                    p_current = {k: jnp.asarray(loaded[k]) for k in loaded.files}
                    start_step = saved_step
        except Exception as e:
            print(f"⚠️ Failed to parse resume trajectory: {e}. Starting from step 0.")

    step_fn = build_capacity_step_fn(cfg, buffers, arm=arm)

    # If start_step == 0, record initial step 0 baseline evaluation
    if start_step == 0 and not trajectory:
        print("\n🔍 Scoring Step 0 Baseline Metrics...")
        init_rec = evaluate_checkpoint_trajectory(
            p_current, base_params, buffers, cfg, tokenizer, arm,
            step_idx=0, train_loss=14.0, wall_clock_sec=0.0,
            disp_tuple=(0.0, 0.0, 0.0), last_grads=None, seq_len=seq_len
        )
        trajectory.append(init_rec)
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        print(f"  [Step   0/{total_steps}] Held-Out NLL: {init_rec['heldout_nll']:.4f} | Retention: {init_rec['retention_nll']:.4f} | Mid Ablation Δ: {init_rec['mid_ablation_delta_nll']:+.4f}")

    checkpoint_milestones = set(CHECKPOINT_STEPS)
    last_grads = None
    t_start = time.time()
    accum_wall_time = trajectory[-1]["wall_clock_sec"] if trajectory else 0.0

    print(f"\n⚡ Commencing training from step {start_step + 1} to {total_steps}...")

    for s in range(start_step, total_steps):
        step_idx = s + 1
        t_step_start = time.time()

        batch_texts = [train_texts[i] for i in stream_indices[s]]
        encoded = tokenizer(batch_texts, padding="max_length", max_length=seq_len, truncation=True, return_tensors="np")
        t_arr = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
        p_arr = jnp.full((t_arr.shape[0], 1), 1, dtype=jnp.int32)

        (loss_val, disp_val), grads = step_fn(p_current, p_arr, t_arr)
        loss_val.block_until_ready()
        disp_val.block_until_ready()
        for g in grads.values():
            g.block_until_ready()

        last_grads = grads
        p_current = apply_arm_parameter_update(p_current, grads, arm=arm, step_idx=step_idx)
        for p in p_current.values():
            p.block_until_ready()

        accum_wall_time += (time.time() - t_step_start)

        if step_idx in checkpoint_milestones:
            d_core = float(disp_val[0])
            d_mid = float(jnp.mean(disp_val[1:12]))
            d_outer = float(jnp.mean(disp_val[12:]))

            rec = evaluate_checkpoint_trajectory(
                p_current, base_params, buffers, cfg, tokenizer, arm,
                step_idx=step_idx, train_loss=float(loss_val),
                wall_clock_sec=accum_wall_time,
                disp_tuple=(d_core, d_mid, d_outer),
                last_grads=last_grads,
                seq_len=seq_len
            )
            trajectory.append(rec)

            with open(traj_path, "w") as f:
                json.dump(trajectory, f, indent=2)

            # Save checkpoint weights
            p_host = jax.device_get(p_current)
            npz_step_path = os.path.join(out_dir, f"{arm}_step_{step_idx}.npz")
            np.savez_compressed(npz_step_path, **p_host)

            print(f"  [Step {step_idx:4d}/{total_steps}] Train: {float(loss_val):.4f} | Held-Out: {rec['heldout_nll']:.4f} | Gap: {rec['gen_gap']:+.4f} | Ret: {rec['retention_nll']:.4f} | AblΔ: {rec['mid_ablation_delta_nll']:+.4f} | D: [C:{d_core:.2f} M:{d_mid:.2f}] | |Δθ|: [C:{rec['delta_theta_core']:.2e} M:{rec['delta_theta_mid']:.2e}]")

        if s % 20 == 0:
            import gc
            gc.collect()

    print(f"\n✓ Completed {arm} ({total_steps} steps). Trajectory saved to {traj_path}")


# ==============================================================================
# 6. TRAJECTORY CROSSOVER ANALYZER
# ==============================================================================

def analyze_crossover(out_dir: str = "experiments/long_horizon_checkpoints"):
    traj_a_path = os.path.join(out_dir, "ArmA_trajectory.json")
    traj_c_path = os.path.join(out_dir, "ArmC_trajectory.json")

    if not os.path.exists(traj_a_path) or not os.path.exists(traj_c_path):
        print("Both ArmA and ArmC trajectories must exist to evaluate crossover.")
        return

    with open(traj_a_path, "r") as f:
        ta = json.load(f)
    with open(traj_c_path, "r") as f:
        tc = json.load(f)

    steps_a = {r["step"]: r for r in ta}
    steps_c = {r["step"]: r for r in tc}
    common_steps = sorted(list(set(steps_a.keys()) & set(steps_c.keys())))

    print("\n" + "=" * 95)
    print("🔬 LONG-HORIZON CAPACITY TRAJECTORY & CROSSOVER ANALYSIS")
    print("=" * 95)
    print(f"{'Step':<6} | {'Arm A Held-Out':<16} | {'Arm C Held-Out':<16} | {'Δ(t) [C - A]':<16} | {'Arm C Ablation Δ':<18} | {'Status'}")
    print("-" * 95)

    consecutive_negative = 0
    crossover_step = None

    for s in common_steps:
        ra = steps_a[s]
        rc = steps_c[s]
        delta = rc["heldout_nll"] - ra["heldout_nll"]
        abl_c = rc["mid_ablation_delta_nll"]

        if delta < 0:
            consecutive_negative += 1
            status = f"✅ Arm C Ahead ({consecutive_negative} in row)"
            if consecutive_negative >= 3 and crossover_step is None:
                crossover_step = s
        else:
            consecutive_negative = 0
            status = "Arm A Ahead" if delta > 0.0005 else "Parity"

        print(f"{s:<6d} | {ra['heldout_nll']:<16.4f} | {rc['heldout_nll']:<16.4f} | {delta:<+16.4f} | {abl_c:<+18.4f} | {status}")

    print("=" * 95)
    if crossover_step is not None:
        print(f"🎉 PERSISTENT CROSSOVER CONFIRMED at Step {crossover_step}!")
        print(f"   Arm C maintained lower held-out NLL for >= 3 consecutive checkpoints.")
    else:
        print("ℹ️ Persistent crossover not yet reached.")


# ==============================================================================
# 7. CLI ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-Horizon Capacity Unlocking Experiment")
    parser.add_argument("--arm", type=str, default="ArmA", choices=["ArmA", "ArmC", "crossover", "run-both"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_small_latest.pt")
    parser.add_argument("--out_dir", type=str, default="experiments/long_horizon_checkpoints")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    # Path resolution
    if not os.path.isabs(args.checkpoint):
        ckpt_path = os.path.join(WORKSPACE_ROOT, args.checkpoint)
    else:
        ckpt_path = args.checkpoint

    if not os.path.isabs(args.out_dir):
        out_dir_path = os.path.join(WORKSPACE_ROOT, args.out_dir)
    else:
        out_dir_path = args.out_dir

    if args.arm == "crossover":
        analyze_crossover(out_dir_path)
    elif args.arm == "run-both":
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        for a in ["ArmA", "ArmC"]:
            cmd = [
                sys.executable,
                __file__,
                "--arm", a,
                "--steps", str(args.steps),
                "--seed", str(args.seed),
                "--seq_len", str(args.seq_len),
                "--checkpoint", ckpt_path,
                "--out_dir", out_dir_path,
            ]
            print(f"\nLaunching subprocess: {' '.join(cmd)}")
            subprocess.check_call(cmd, env=env)

        analyze_crossover(out_dir_path)
    else:
        run_arm_capacity_training(
            arm=args.arm,
            total_steps=args.steps,
            seed=args.seed,
            seq_len=args.seq_len,
            checkpoint_path=ckpt_path,
            out_dir=out_dir_path,
            resume=not args.no_resume,
        )
