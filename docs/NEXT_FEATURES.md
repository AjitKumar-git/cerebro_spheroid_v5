# CerebroSpheroid V5: Next Features & Backlog

Planned improvements tracked for future development cycles (do not enable mid-run without updating auto-tuner and multi-TPU shape contracts).

---

## 1. Zero-Waste Sequence Packing & Longer Prompt Windows (`dense_edu` / `conversation_sft` / `reasoning_anneal`)

- **Problem with current fixed `(prompt_len=32, seq_len=256)`**:
  - In [`ElasticBucketBatcher.next_batch`](../src/elastic_window.py), prompts longer than `32` tokens are hard-truncated via `q_ids[:bp]`, losing instructions on longer Cosmopedia/OpenHermes/GSM8K prompts.
  - Short responses (`< 256` tokens) and short prompts (`< 32` tokens) are padded with `-100` / `pad_id` rather than packed with another example.
  - On long articles, `sample_window` samples one random `256`-token slice and discards remaining windows from that document.
- **Recommended Implementation**:
  1. **Multi-example sequence packing into a single static shape** (zero extra XLA recompilation, 100% compatible with 8-TPU `pmap` Local-SGD) so short Q&As fill the full window with near-zero `-100` padding waste.
  2. **Sequential multi-window extraction** from long documents (emitting consecutive windows with preceding context token handoff instead of discarding most of the article).
  3. **Step prompt capacity up** (`prompt_len`: `32` -> `64` or `128`) before instruction/reasoning phases (`conversation_sft`, `reasoning_anneal`) and re-run `--auto-tune` for that new static shape.

---

## 2. Multi-TPU Safe `--elastic_window` (Round-Level Bucketing)

- **Why it is currently disabled under Multi-TPU**:
  - [`scripts/train_v5_tpu_jax.py`](../scripts/train_v5_tpu_jax.py) force-disables `--elastic_window` when `use_multi_tpu_local_sgd` is active because `ElasticBucketBatcher.next_batch()` pops mixed bucket shapes across consecutive pulls, whereas `pmap` + `lax.scan` requires all `num_devices * local_steps` shards in a round to share one uniform `(bp, bt)` shape.
  - Default `--total_window 288` falls below `512` in [`get_default_elastic_buckets`](../src/elastic_window.py), collapsing to a single `[("balanced", 144, 144)]` bucket.
  - `run_auto_tuner` currently benchmarks only the fixed `(prompt_len, seq_len)` (`32, 256`), so enabling larger buckets (`512` or `1024` tokens) would risk HBM OOM under batch sizes tuned for `288` tokens.
- **Required Changes to Enable Safely**:
  1. **Round-level homogeneous bucketing**: Add a `round_items` / batch-group pull mode in `ElasticBucketBatcher` so every shard in an `(eff_num_devices, eff_local_steps, B, L)` round comes from the same bucket `b_idx`.
  2. **Worst-case bucket auto-tuning**: Run `run_auto_tuner` against the largest `(max_bp, max_bt)` bucket shape in `buckets` instead of `(32, 256)` so `batch_size` and `remat` never OOM when that bucket triggers.
  3. **Window budget & positional bounds**: Set `--total_window` to `512` or `1024` while enforcing `max_bp + max_bt <= 1024` (`ModelConfigV5.max_phase_steps` in [`src/jax/v5_model.py`](../src/jax/v5_model.py)), and expect a brief adaptation window for outer positional/phase embeddings (`positions >= 288`) when migrating from `(32, 256)` checkpoints.
