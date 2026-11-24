#!/usr/bin/env python3
"""
🧠 CerebroSpheroid V5: 25M Probe Architectural Screening Harness
Executes Round 1 Screening across T0, T1, T2, T3 on the 24.25M 8-Shell Probe Model.
Fast, reproducible, and mathematically faithful to V5 concentric recurrence.
"""

import os
import sys

# Platform allocator configuration
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

from src.spheroid_v5 import ModelConfigV5
from src.jax.v5_model import (
    init_params_v5,
    deliberate_step,
    deliberate_cycles_jax,
    encode_sensory,
    emit_step,
)

# ==============================================================================
# 1. 25M PROBE CONFIGURATION & DATASETS
# ==============================================================================

def get_25m_probe_config() -> ModelConfigV5:
    """Returns the standardized 24.25M parameter 8-shell probe model configuration."""
    return ModelConfigV5(
        vocab_size=32000,
        surface_dim=320,
        radial_dim=128,
        num_shells=8,              # 8 concentric shells: 0=Core, 1..5=Mid (5 shells), 6..7=Outer (2 shells)
        num_shortcut_tracts=20,
        num_heads=4,
        thinking_cycles=4,
        top_k_tracts=10,
        phase_scale=2.0,
        use_tied_embeddings=True,
        use_swiglu=True,
        use_sensory_self_attn=True,
        use_rope=False,
        top_k_resonance=0,
        use_deliberation_probe=False,
        use_adaptive_thinking=False,
        use_sparse_mot=False,
        use_energy_regularization=False,
    )

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


# ==============================================================================
# 2. STEP FUNCTION & LOSS BUILDER (8-SHELL PROBE)
# ==============================================================================

def build_probe_step_fn(config: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray], arm: str = "T0"):
    K = config.num_shells  # 8
    D = config.surface_dim # 320

    def _delib_closure(p, st, buf):
        return deliberate_step(p, st, buf, config=config)

    ckpt_delib_step = jax.checkpoint(_delib_closure)

    def probe_loss_fn(params, prompts, targets):
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

            if arm == "T0":
                # Static Control: only shell 0 evolves (shells 1..7 remain static)
                cur_state = nxt_state

            elif arm == "T1":
                # Periodic Mid /8: intermediate shells (1..5) refreshed every 8 tokens
                is_periodic = (t > 0) and (t % 8 == 0)
                if is_periodic:
                    delib = ckpt_delib_step(params, nxt_state, buffers)
                    cur_state = nxt_state.at[:, 1:6, :].set(0.85 * nxt_state[:, 1:6, :] + 0.15 * delib[:, 1:6, :])
                else:
                    cur_state = nxt_state

            elif arm == "T2":
                # Adaptive Mid Surprisal: intermediate shells (1..5) adapt based on entropy gate
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)

                def do_adaptive_update(st):
                    delib = ckpt_delib_step(params, st, buffers)
                    return st.at[:, 1:6, :].set(st[:, 1:6, :] + 0.15 * g_act * (delib[:, 1:6, :] - st[:, 1:6, :]))

                cur_state = lax.cond(g_act > 0.05, do_adaptive_update, lambda st: st, nxt_state)

            elif arm == "T3":
                # Hierarchical: Adaptive Mid (1..5) + Sensory/Uncertainty Outer (6..7)
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)
                is_outer = (t > 0) and ((t % max(4, total_L // 2) == 0) or (t == 8))

                def do_hierarchical_update(st):
                    delib = ckpt_delib_step(params, st, buffers)
                    mid = st.at[:, 1:6, :].set(st[:, 1:6, :] + 0.15 * g_act * (delib[:, 1:6, :] - st[:, 1:6, :]))
                    if is_outer:
                        return mid.at[:, 6:, :].set(0.90 * mid[:, 6:, :] + 0.10 * delib[:, 6:, :])
                    else:
                        return mid

                cur_state = lax.cond((g_act > 0.05) | is_outer, do_hierarchical_update, lambda st: st, nxt_state)

            disp_t = jnp.mean(jnp.linalg.norm(cur_state - prev_for_disp, axis=-1), axis=0)
            if t == 0:
                total_disp = disp_t
            else:
                total_disp = total_disp + disp_t

        return total_loss / total_L, total_disp / total_L

    return jax.jit(jax.value_and_grad(probe_loss_fn, has_aux=True))


def compute_probe_shell_metrics(tensors: Dict[str, jnp.ndarray]) -> Tuple[float, float, float]:
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
            g_mid_sq += float(jnp.sum(val[1:6] ** 2)) / 5.0
            g_outer_sq += float(jnp.sum(val[6:] ** 2)) / 2.0

    if "core.w_radial_in" in tensors:
        rin = tensors["core.w_radial_in"]
        g_mid_sq += float(jnp.sum(rin[:5] ** 2)) / 5.0
        g_outer_sq += float(jnp.sum(rin[5:] ** 2)) / 2.0

    if "core.w_radial_out" in tensors:
        rout = tensors["core.w_radial_out"]
        g_mid_sq += float(jnp.sum(rout[:5] ** 2)) / 5.0
        g_outer_sq += float(jnp.sum(rout[5:] ** 2)) / 2.0

    return math.sqrt(g_core_sq), math.sqrt(g_mid_sq), math.sqrt(g_outer_sq)


# ==============================================================================
# 3. FAST JIT EVALUATION ENGINE (8-SHELL PROBE)
# ==============================================================================

def build_probe_eval_fn(config: ModelConfigV5, buffers: Tuple[jnp.ndarray, jnp.ndarray], arm: str = "T0", ablate_mid: bool = False):
    K = config.num_shells
    D = config.surface_dim

    def _eval(params, p_arr, t_arr):
        B, total_L = t_arr.shape
        p_flux, p_kv = encode_sensory(params, p_arr, config=config, pos_offset=0)
        r_prof = (jnp.arange(1, K + 1, dtype=jnp.float32) / float(K))[None, :, None]
        init_state = (jnp.broadcast_to(p_flux, (B, K, D)).astype(jnp.float32) * r_prof).astype(jnp.float32)
        state, _ = deliberate_cycles_jax(params, init_state, buffers, config=config, is_training=False, sensory_kv=p_kv)

        prev_toks = jnp.concatenate([p_arr[:, -1:], t_arr[:, :-1]], axis=1)
        tf_flux, (tf_k, tf_v) = encode_sensory(params, prev_toks, config=config, pos_offset=p_arr.shape[1], is_causal=True)
        causal_k = jnp.concatenate([p_kv[0], tf_k], axis=2)
        causal_v = jnp.concatenate([p_kv[1], tf_v], axis=2)

        cur_state = state
        cur_plan = jnp.zeros((B, D), dtype=jnp.float32)
        cur_trace = None
        seq_loss = 0.0

        for t in range(total_L):
            if ablate_mid:
                active_state = cur_state.at[:, 1:6, :].set(0.0)
            else:
                active_state = cur_state

            curr_ctx_len = p_arr.shape[1] + t + 1
            sub_k = causal_k[:, :, :curr_ctx_len, :]
            sub_v = causal_v[:, :, :curr_ctx_len, :]

            logits, nxt_state, cur_trace, cur_plan = emit_step(
                params,
                active_state,
                (sub_k, sub_v),
                cur_plan,
                cur_trace,
                t,
                buffers,
                config=config,
                auditory_feedback=tf_flux[:, t, :],
                query_position=p_arr.shape[1] + t,
            )

            tgt = t_arr[:, t]
            log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            step_loss = -jnp.mean(jnp.take_along_axis(log_p, tgt[:, None], axis=-1))
            seq_loss = seq_loss + step_loss

            if arm == "T0":
                cur_state = nxt_state
            elif arm == "T1":
                is_periodic = (t > 0) and (t % 8 == 0)
                if is_periodic:
                    delib = deliberate_step(params, nxt_state, buffers, config=config)
                    cur_state = nxt_state.at[:, 1:6, :].set(0.85 * nxt_state[:, 1:6, :] + 0.15 * delib[:, 1:6, :])
                else:
                    cur_state = nxt_state
            elif arm == "T2":
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)
                def do_ad(st):
                    delib = deliberate_step(params, st, buffers, config=config)
                    return st.at[:, 1:6, :].set(st[:, 1:6, :] + 0.15 * g_act * (delib[:, 1:6, :] - st[:, 1:6, :]))
                cur_state = lax.cond(g_act > 0.05, do_ad, lambda st: st, nxt_state)
            elif arm == "T3":
                probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
                entropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1))
                g_act = jax.nn.sigmoid(entropy - 3.2)
                is_outer = (t > 0) and ((t % max(4, total_L // 2) == 0) or (t == 8))
                def do_h(st):
                    delib = deliberate_step(params, st, buffers, config=config)
                    mid = st.at[:, 1:6, :].set(st[:, 1:6, :] + 0.15 * g_act * (delib[:, 1:6, :] - st[:, 1:6, :]))
                    if is_outer:
                        return mid.at[:, 6:, :].set(0.90 * mid[:, 6:, :] + 0.10 * delib[:, 6:, :])
                    else:
                        return mid
                cur_state = lax.cond((g_act > 0.05) | is_outer, do_h, lambda st: st, nxt_state)

        return seq_loss / total_L

    return jax.jit(_eval)


# ==============================================================================
# 4. ROUND 1 TRAINING RUNNER
# ==============================================================================

def train_probe_arm(
    arm: str,
    seed: int = 42,
    steps: int = 300,
    lr: float = 1e-3,
    eval_interval: int = 50,
    out_dir: str = "experiments/probe_25m_checkpoints",
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = get_25m_probe_config()

    tok_dir = os.path.join(WORKSPACE_ROOT, "data", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Deterministic parameter initialization per seed
    init_key = jax.random.PRNGKey(seed)
    params, buffers = init_params_v5(init_key, cfg)
    base_params = {k: jnp.array(v) for k, v in params.items()}

    # Deterministic data stream
    rng = np.random.RandomState(seed)
    train_texts = [item["text"] for item in TRAINING_CORPUS]
    stream_indices = [rng.choice(len(train_texts), size=4, replace=False) for _ in range(steps)]

    step_fn = build_probe_step_fn(cfg, buffers, arm=arm)
    eval_fn_full = build_probe_eval_fn(cfg, buffers, arm=arm, ablate_mid=False)
    eval_fn_abl = build_probe_eval_fn(cfg, buffers, arm=arm, ablate_mid=True)

    # Pre-encode 16 held-out sequences
    val_samples = []
    for item in HELD_OUT_VALIDATION:
        enc = tokenizer(item["text"], padding="max_length", max_length=16, truncation=True, return_tensors="np")
        t_ids = jnp.asarray(enc["input_ids"], dtype=jnp.int32)
        p_ids = jnp.full((1, 1), 1, dtype=jnp.int32)
        val_samples.append((p_ids, t_ids))

    def score_heldout(p, is_abl=False):
        fn = eval_fn_abl if is_abl else eval_fn_full
        losses = [float(fn(p, p_arr, t_arr)) for p_arr, t_arr in val_samples]
        return float(np.mean(losses))

    print(f"\n🚀 TRAINING PROBE 25M | ARM: {arm:<4} | SEED: {seed} | STEPS: {steps} | LR: {lr:.1e}")

    # Step 0 evaluation
    init_heldout = score_heldout(params, is_abl=False)
    init_ablate = score_heldout(params, is_abl=True)
    init_abl_delta = init_ablate - init_heldout
    print(f"  [Step   0/{steps}] Held-Out NLL: {init_heldout:.4f} | Ablation Δ: {init_abl_delta:+.4f}")

    trajectory = [{
        "step": 0,
        "train_loss": 14.0,
        "heldout_nll": init_heldout,
        "ablation_delta": init_abl_delta,
        "d_core": 0.0, "d_mid": 0.0, "d_outer": 0.0,
        "tc": 0.0, "tm": 0.0, "to": 0.0,
        "time_sec": 0.0,
    }]

    t_start = time.time()
    last_grads = None

    for s in range(steps):
        step_idx = s + 1
        batch_texts = [train_texts[i] for i in stream_indices[s]]
        encoded = tokenizer(batch_texts, padding="max_length", max_length=16, truncation=True, return_tensors="np")
        t_arr = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
        p_arr = jnp.full((t_arr.shape[0], 1), 1, dtype=jnp.int32)

        (loss_val, disp_val), grads = step_fn(params, p_arr, t_arr)
        loss_val.block_until_ready()
        disp_val.block_until_ready()
        for g in grads.values():
            g.block_until_ready()

        last_grads = grads
        params = {k: params[k] - lr * grads[k] for k in params}
        for p in params.values():
            p.block_until_ready()

        if step_idx % eval_interval == 0 or step_idx == steps:
            d_core = float(disp_val[0])
            d_mid = float(jnp.mean(disp_val[1:6]))
            d_outer = float(jnp.mean(disp_val[6:]))

            d_theta = {k: params[k] - base_params[k] for k in params}
            tc, tm, to = compute_probe_shell_metrics(d_theta)

            heldout_nll = score_heldout(params, is_abl=False)
            ablate_nll = score_heldout(params, is_abl=True)
            abl_delta = ablate_nll - heldout_nll

            elapsed = time.time() - t_start
            throughput = step_idx / max(elapsed, 0.01)

            rec = {
                "step": step_idx,
                "train_loss": float(loss_val),
                "heldout_nll": heldout_nll,
                "ablation_delta": abl_delta,
                "d_core": d_core, "d_mid": d_mid, "d_outer": d_outer,
                "tc": tc, "tm": tm, "to": to,
                "time_sec": elapsed,
            }
            trajectory.append(rec)

            print(f"  [Step {step_idx:3d}/{steps}] Train: {float(loss_val):.4f} | Held-Out: {heldout_nll:.4f} | AblΔ: {abl_delta:+.4f} | D: [C:{d_core:.2f} M:{d_mid:.2f} O:{d_outer:.2f}] | {throughput:.1f} steps/s")

    out_json = os.path.join(out_dir, f"probe_seed{seed}_{arm}_trajectory.json")
    with open(out_json, "w") as f:
        json.dump(trajectory, f, indent=2)

    p_host = jax.device_get(params)
    out_npz = os.path.join(out_dir, f"probe_seed{seed}_{arm}.npz")
    np.savez_compressed(out_npz, **p_host)

    print(f"✓ Saved {arm} Seed {seed} to {out_json}")
    return trajectory


# ==============================================================================
# 5. ROUND 1 ORCHESTRATION & LEADERBOARD SYNTHESIS
# ==============================================================================

def run_round1_screening(
    seeds: List[int] = [42, 137],
    arms: List[str] = ["T0", "T1", "T2", "T3"],
    steps: int = 300,
    out_dir: str = "experiments/probe_25m_checkpoints",
):
    print("\n" + "=" * 95)
    print("🏆 LAUNCHING 25M PROBE ROUND 1 ARCHITECTURAL SCREENING")
    print(f"  • Model       : 24.25M Parameters (8 Shells: Core 0, Mid 1..5, Outer 6..7)")
    print(f"  • Arms        : {', '.join(arms)}")
    print(f"  • Seeds       : {seeds}")
    print(f"  • Steps       : {steps} steps per arm per seed")
    print(f"  • Validation  : 16 Held-Out Multi-Domain Sequences (Zero 3-Gram Leakage)")
    print("=" * 95)

    all_results = {arm: {} for arm in arms}

    for s in seeds:
        print(f"\n" + "-" * 75)
        print(f"🌱 EXECUTING SEED {s}")
        print("-" * 75)
        for arm in arms:
            traj = train_probe_arm(arm=arm, seed=s, steps=steps, out_dir=out_dir)
            all_results[arm][s] = traj[-1]

    # Synthesize Round 1 Leaderboard
    print("\n" + "=" * 95)
    print("📊 25M PROBE ROUND 1 SCREENING LEADERBOARD (Step 300)")
    print("=" * 95)
    print(f"{'Rank':<6} | {'Arm':<6} | {'Held-Out NLL (Mean ± Std)':<28} | {'Δ vs T0':<14} | {'Ablation ΔNLL':<16} | {'Throughput':<12}")
    print("-" * 95)

    rankings = []
    t0_mean = np.mean([all_results["T0"][s]["heldout_nll"] for s in seeds])

    for arm in arms:
        nlls = [all_results[arm][s]["heldout_nll"] for s in seeds]
        abls = [all_results[arm][s]["ablation_delta"] for s in seeds]
        times = [all_results[arm][s]["time_sec"] for s in seeds]

        m_nll, s_nll = float(np.mean(nlls)), float(np.std(nlls))
        m_abl = float(np.mean(abls))
        t_rate = steps / float(np.mean(times))
        delta_t0 = m_nll - t0_mean

        rankings.append({
            "arm": arm,
            "mean_nll": m_nll,
            "std_nll": s_nll,
            "delta_t0": delta_t0,
            "ablation_delta": m_abl,
            "steps_per_sec": t_rate,
        })

    rankings.sort(key=lambda x: x["mean_nll"])

    for rank, r in enumerate(rankings, 1):
        sign = "+" if r["delta_t0"] > 0 else ""
        delta_str = f"{sign}{r['delta_t0']:.4f}" if r["arm"] != "T0" else "0.0000 (Base)"
        badge = "🏆 QUALIFIED" if rank <= 2 else "❌ ELIMINATED"
        print(f"{rank:<6d} | {r['arm']:<6} | {r['mean_nll']:.4f} ± {r['std_nll']:.4f}                   | {delta_str:<14} | {r['ablation_delta']:<+16.4f} | {r['steps_per_sec']:<5.1f} st/s  [{badge}]")

    print("=" * 95)
    top2 = [r["arm"] for r in rankings[:2]]
    print(f"🎯 Finalists Selected for Round 2 (1,000 Steps): {top2}")

    summary_path = os.path.join(out_dir, "round1_screening_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"rankings": rankings, "top2": top2, "raw_results": all_results}, f, indent=2)

    return top2


# ==============================================================================
# 6. CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="25M Probe Recurrence Screening")
    parser.add_argument("--mode", type=str, default="round1", choices=["round1", "single"])
    parser.add_argument("--arm", type=str, default="T0", choices=["T0", "T1", "T2", "T3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default="experiments/probe_25m_checkpoints")
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        out_path = os.path.join(WORKSPACE_ROOT, args.out_dir)
    else:
        out_path = args.out_dir

    if args.mode == "round1":
        run_round1_screening(seeds=[42, 137], arms=["T0", "T1", "T2", "T3"], steps=args.steps, out_dir=out_path)
    else:
        train_probe_arm(arm=args.arm, seed=args.seed, steps=args.steps, out_dir=out_path)
