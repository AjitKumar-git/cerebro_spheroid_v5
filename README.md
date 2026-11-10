# 🧠 CerebroSpheroid V5: Deep Concentric Laminae & Recurrent Deliberation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework: Pure JAX / PyTorch](https://img.shields.io/badge/Framework-JAX%20%7C%20PyTorch-red.svg)](https://jax.readthedocs.io/)
[![Architecture: Concentric Spheroid](https://img.shields.io/badge/Architecture-Concentric_Spheroid_V5-green.svg)]()

CerebroSpheroid V5 is a recurrent cognitive neural architecture designed as a high-efficiency alternative to standard Transformer decoders. It replaces quadratic KV caches with a **fixed $O(1)$ concentric spherical state**, provides **true latent subconscious deliberation**, and uses **Riemannian Polar Quintic Newton-Schulz guards ($\rho \le 1.0$)** to guarantee contractive mathematical stability.

---

## ⚡ Key Upgrades in V5 over V4

1. **Affine-Tied Sensory-Motor Manifold**: Eliminates 30M+ redundant lookup parameters ($W_{\text{motor}} = \gamma \odot W_{\text{sensory}}$), dedicating >90% of parameters to active volumetric reasoning.
2. **Deep Concentric Laminae (32 to 48 Shells)**: Expands radial wave propagation depth for richer hierarchical abstraction.
3. **SwiGLU Gated Wave Equations**: Introduces non-linear SwiGLU projections across intra-shell and inter-shell convection channels.
4. **8-Head Resonant Arcuate Fasciculus**: Multi-head cross-resonance between motor emission and sensory prompts with temporal recency biasing.
5. **Native Normalized Phase Clock ($p_{\text{scaled}} = 2.0$)**: Eliminates static attractor basin traps from step 0.

---

## 📐 Configurations

| Model | Total Parameters | Active Compute Parameters | Shells ($K$) | Surface Dim ($D$) | Radial Dim ($D_r$) | State Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V5-Compact** | **250.8M** | **226.2M (90.2%)** | 32 | 768 | 512 | **49.2 KB** |
| **V5-Base** | **512.4M** | **475.6M (92.8%)** | 48 | 1024 | 768 | **98.3 KB** |

---

## 📂 Directory Structure

```
cerebro_spheroid_v5/
├── configs/                  # V5 Model preset JSON specifications
│   ├── v5_compact_250m.json
│   └── v5_base_500m.json
├── data/                     # Curated reasoning datasets
│   └── cerebro_cognitive_8k/
├── src/                      # Core model implementations
│   ├── spheroid_v5.py        # PyTorch reference implementation
│   ├── jax/                  # High-throughput JAX PJRT engine
│   │   └── v5_model.py
│   ├── corpus.py             # LLaMA Byte-Fallback tokenizer
│   ├── elastic_window.py     # Elastic context batcher
│   └── cloud_sync.py         # Asynchronous checkpoint sync
├── training/                 # Training engines & acceptance tests
│   └── test_v5_architecture.py
└── checkpoints/              # Model weights storage
```
