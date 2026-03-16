# Autoresearch Program: Architecture Improvements

## Overview

You are autonomously improving a specific architecture's velocity network to generate better 3D hard sphere configurations. You will iteratively edit the model code, test changes, and keep what works.

**Architecture changes are isolated**: you edit one model file, test only that architecture. Flow matching code is frozen during this phase.

## Editable Files

You may **only** edit **one** architecture file per session:
- `models/equiv_gnn.py` — Equivariant GNN (PaiNN-based)
- `models/transformer.py` — Transformer with distance bias
- `models/pairformer.py` — Pairformer with triangular updates

Pick one architecture at the start of the session and stick with it.

## Frozen Files (DO NOT EDIT)

Everything else is frozen: `experiments/`, `metrics/`, `data/`, `autoresearch/`, `flow_matching/`, and the other model files.

## Interface Constraint

The model's forward signature **must be preserved**:

```python
def forward(self, positions: Tensor, t: Tensor) -> Tensor:
    """
    Args:
        positions: (batch, N, 3) atom positions
        t: (batch,) timesteps in [0, 1]
    Returns:
        velocities: (batch, N, 3) predicted velocity field
    """
```

The constructor may accept any kwargs, but must work with the existing `SIZE_PRESETS` and `MODEL_DEFAULTS` from `experiments/model_registry.py`. You may add new kwargs with defaults.

## Metric

- **Primary:** `gr_distance` — L1 distance between generated and ground-truth g(r). Lower is better.
- **Secondary:** `clash_rate` — reported for interpretability.
- **Baseline:** Run `uv run python autoresearch/baseline.py --data hard_sphere_N50` first.

## Running the Harness

```bash
# Test a single architecture (all 8 GPUs run variants of that arch)
uv run python autoresearch/run.py --archs transformer --data hard_sphere_N50 --n_gpus 8

# Longer runs if needed to see architecture effects
uv run python autoresearch/run.py --archs transformer --data hard_sphere_N50 --train_time 15 --n_gpus 8

# Quick single-GPU test during development
uv run python autoresearch/run.py --archs transformer --data hard_sphere_N50 --train_time 5 --n_gpus 1
```

Each variant trains for `--train_time` minutes (default 10). Use longer times (15 min) for architecture changes that need more training to show effect.

**Note:** Use `--archs <single_arch>` for architecture experiments (not `--archs all` — that's for flow matching changes).

## Workflow

1. **Read this file** and `outputs/autoresearch/experiments.jsonl`
2. **Read the target architecture file** (e.g., `models/transformer.py`)
3. **Hypothesize** a specific change
4. **Implement** the change
5. **Test:** `uv run python autoresearch/run.py --archs <target> --data hard_sphere_N50 --n_gpus 8`
6. **Parse** JSON output
7. **If improved** → `git add models/<target>.py && git commit -m "autoresearch: <description>"`
   **If not** → `git checkout -- models/<target>.py`
8. **Log** using `autoresearch/experiment_log.py`
9. **Repeat**

## Seeded Ideas

### General (applicable to all architectures)

1. **Activation function** — Try SiLU/Swish instead of ReLU/GELU. Or GELU instead of SiLU.
2. **Normalization** — Try RMSNorm instead of LayerNorm (faster, sometimes better).
3. **Timestep conditioning** — AdaLN (adaptive layer norm) vs concatenation vs addition. Try FiLM conditioning.
4. **Residual connections** — Pre-norm vs post-norm. Residual scaling (alpha * residual + x).
5. **Feature initialization** — How atom features are initialized from positions. Try learnable embeddings + position encoding.

### GNN-specific (equiv_gnn.py)

6. **Message aggregation** — Mean vs sum vs attention-weighted aggregation.
7. **Edge features** — Richer distance encoding (Bessel functions, polynomial basis).
8. **Vector feature updates** — Different coupling between scalar and vector channels.
9. **Cutoff function** — Cosine cutoff vs polynomial vs envelope function.

### Transformer-specific (transformer.py)

10. **Attention bias** — Different ways to inject pairwise distance into attention (additive vs multiplicative).
11. **Position encoding** — Sinusoidal vs RoPE vs learned.
12. **MLP structure** — GLU variants (SwiGLU, GeGLU).
13. **Attention scaling** — QK normalization, different temperature.

### Pairformer-specific (pairformer.py)

14. **Triangular update order** — Outgoing before incoming vs alternating.
15. **Pair representation init** — Richer initial pair features beyond just distance.
16. **Single-pair coupling** — How information flows between single and pair tracks.
17. **Gating mechanisms** — Different gating in triangular updates.

## Tips

- Make **one change at a time**.
- Architecture changes sometimes need more training time to show effect. If a promising change shows no effect at 10 min, try `--train_time 15`.
- Watch the **per-variant breakdown**: a change that only helps large models suggests it's about capacity; one that only helps small models suggests better inductive bias.
- Don't change `experiments/model_registry.py` — if your change needs new constructor args, add them with defaults so existing presets still work.
- The experiment log is your memory. Read it before each iteration.
