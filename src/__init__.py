"""
CerebroSpheroid V5: High-Performance Cortical Architecture
==========================================================
Unified functional namespace providing modular isolation across core subsystems:

1. MODEL & COGNITIVE ARCHITECTURE (`src.model` / `spheroid_v5`, `cortical_expansion`, `cerebro_reward_model`)
   - Concentric cortical shells, radial manifold dynamics, and recurrent latent deliberation cycles.
   - Net2Net cortical expansion with parameter inheritance.
   - Cognitive reasoning and exploration reward verification.

2. OPTIMIZATION & MANIFOLD REGULARIZATION (`src.optimizer` / `spheroid_optimizer`, `src.jax.scmer_optimizer`)
   - SCMER (Selective Cosine-Margin Embedding Regularization) with Polar Order-5 Newton-Schulz orthogonalization.
   - Spectral norm clamping and rank-adaptive gating acceleration.
   - Native JAX TPU SCMER and optax training step builders.

3. DATA & CONTEXT CURRICULUM (`src.data` / `rectified_streamer`, `corpus`, `elastic_window`, `graded_curriculum`)
   - 100% loss-bearing contiguous pretraining data streamer.
   - SmolLM2 4-stage developmental context curriculum (512 -> 1024 -> 2048 -> 4096 tokens).
   - Multi-corpus streaming abstractions (FineWeb, The Stack v2, Cosmopedia v2).
   - Elastic context window scheduling and teacher distillation digestion.

4. CLOUD SYNCHRONIZATION & REMOTE VAULT (`src.cloud` / `cloud_sync`, `hf_utils`, `src.jax.checkpoint`)
   - Sequential chunked streaming checkpoint mirror to bioflo.in (Cloudflare 100MB safe).
   - 0ms TPU pause asynchronous checkpoint management (bare BF16 remote, full local state).
   - Centralized website vault credential management and authenticated HuggingFace streaming.

5. AGENTIC & REASONING INFRASTRUCTURE (`src.agentic` / `agentic_harness`, `factual_knowledge_engine`)
   - Standardized 3-tier reasoning boundary (<think> -> <tool_call> -> response).
   - Factual table grounding and knowledge retrieval.
"""

# Functional Subsystem Aliases / Namespaces
from . import cloud_sync as cloud
from . import rectified_streamer as streamer
from . import corpus
from . import elastic_window
from . import hf_utils
from . import spheroid_v5 as model
from . import spheroid_optimizer as optimizer
from . import agentic_harness as agentic
from . import jax

# Primary Functional Exports
from .cloud_sync import (
    push_checkpoint,
    push_checkpoint_async,
    pull_checkpoint,
    get_sync_token,
    notify_training_start,
    push_remote_token_tracker,
    fetch_remote_token_tracker,
    clear_remote_tokens,
)
from .hf_utils import (
    get_hf_token,
    fetch_hf_token_from_website,
)
from .rectified_streamer import (
    RectifiedContiguousPretrainStreamer,
)
from .spheroid_optimizer import (
    SCMEROptimizer,
    newton_schulz5,
    build_scmer_optimizer,
)
from .cortical_expansion import (
    build_cortical_expansion_masks,
    apply_net2net_cortical_init,
    generate_expanded_tract_wiring,
    count_cortical_expansion_params,
    compute_cortical_anneal_factor,
)
from .cerebro_reward_model import (
    CerebroExplorationRewardModel,
    CerebroCuriosityRND,
    ConcentricManifoldExplorer,
)
from .agentic_harness import (
    AgenticSpheroidHarness,
    SandboxedToolExecutor,
    execute_agentic_tool,
)
from .factual_knowledge_engine import (
    FactualKnowledgeEngine,
)

__all__ = [
    # Namespaces
    "cloud",
    "streamer",
    "corpus",
    "elastic_window",
    "hf_utils",
    "model",
    "optimizer",
    "agentic",
    "jax",
    # Functions & Classes
    "push_checkpoint",
    "push_checkpoint_async",
    "pull_checkpoint",
    "get_sync_token",
    "notify_training_start",
    "push_remote_token_tracker",
    "fetch_remote_token_tracker",
    "clear_remote_tokens",
    "get_hf_token",
    "fetch_hf_token_from_website",
    "RectifiedContiguousPretrainStreamer",
    "SCMEROptimizer",
    "newton_schulz5",
    "build_scmer_optimizer",
    "build_cortical_expansion_masks",
    "apply_net2net_cortical_init",
    "generate_expanded_tract_wiring",
    "count_cortical_expansion_params",
    "compute_cortical_anneal_factor",
    "CerebroExplorationRewardModel",
    "CerebroCuriosityRND",
    "ConcentricManifoldExplorer",
    "AgenticSpheroidHarness",
    "SandboxedToolExecutor",
    "execute_agentic_tool",
    "FactualKnowledgeEngine",
]
