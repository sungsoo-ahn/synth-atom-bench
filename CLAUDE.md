# CLAUDE.md

## Project: Scaling Laws for Structure Generation

### Motivation

Scaling laws are well-established for predictive atomistic models (UMA, Nomura et al., DPA-3). No one has studied whether similar scaling laws hold for **generative models** that produce atomic structures. This project fills that gap with synthetic benchmarks where the ground truth distribution is fully known.

### Setup: Model Polymer Chain

A linear chain of N atoms (N = 10–20) in 3D, connected as 1-2-3-...-N. A coarse-grained polymer analogous to a protein backbone. The fixed chain topology eliminates ambiguity about which atoms interact.

The energy function sums terms with increasing geometric complexity:

- **k=2 (bond lengths):** V₂ = Σᵢ k₂(rᵢ,ᵢ₊₁ - r₀)². Preferred distance between consecutive atoms.
- **k=3 (bond angles):** V₃ = Σᵢ k₃(θᵢ₋₁,ᵢ,ᵢ₊₁ - θ₀)². Preferred angle at each atom, analogous to Ramachandran constraints.
- **k=4 (dihedrals):** V₄ = Σᵢ k₄(φᵢ₋₁,ᵢ,ᵢ₊₁,ᵢ₊₂ - φ₀)². Torsion angles introducing longer-range structural coupling.

Terms are cumulative: k=3 includes bond+angle, k=4 includes all three. Simple harmonic potentials throughout — the only thing that changes across k is the geometric nature of the constraint.

Three fixed presets:
- `multibody_2`: V₂ only
- `multibody_23`: V₂ + V₃
- `multibody_234`: V₂ + V₃ + V₄

### Data Generation

Samples from the Boltzmann distribution p(x) ∝ exp(-E(x)/T) via overdamped Langevin dynamics with parallel chains on GPU. Three temperatures per preset:
- T_high (T=2.0): Broad, fluid-like. Easy to generate.
- T_mid (T=1.0): Moderate structure. Intermediate difficulty.
- T_low (T=0.5): Sharp distribution near minima. Hard to generate.

Training datasets of varying sizes (10K, 50K, 200K) for data scaling.

```bash
uv run python data/generate_multibody.py \
  --N 20 --T 1.0 --preset multibody_23 \
  --num_samples 50000 --seed 42 \
  --output outputs/data/multibody_23_N20_T1.0/train.npz
```

Output .npz fields: `positions` (num_samples, N, 3), `box_size`, `N`, `temperature`, `preset`, potential params (`k2`, `r0`, `k3`, `theta0`, `k4`, `phi0`), `energies`.

## Generative Framework

Conditional flow matching (OT-CFM). DiT (Diffusion Transformer) backbone as velocity network v_θ(x_t, t) → predicted velocity field.

Interpolation: x_t = (1 - t) ε + t x₀, where ε ~ N(0, I), t ∈ [0, 1]
Loss: ||v_θ(x_t, t) - (x₀ - ε)||²
Sampling: ODE integration from x₀ ~ N(0, I) to x₁

## Architecture

**Transformer (DiT-style)** — the sole architecture, in `models/transformer.py`:
- Global self-attention over all atoms
- Pairwise distance features injected as attention bias
- Sinusoidal timestep embedding via adaptive layer norm (adaLN-Zero)
- Random rotation augmentation (learns rotational invariance from data)
- Atom ordering embeddings for chain topology
- Reference: SimpleFold (Apple, 2025) — standard transformer blocks + flow matching
- Model sizes: xs (32d/2L), small (64d/3L), medium (128d/6L), large (256d/8L), xl (384d/10L)

## Metric

**Energy Wasserstein distance** (W₁): Wasserstein-1 distance between the energy distributions of generated samples and reference data. Lower is better.

```python
from metrics.energy import energy_metrics_batched
# Returns: {"energy_wasserstein": float, "mean_energy_ratio": float}
metrics = energy_metrics_batched(generated_samples, dataset)
```

This is the sole metric for training eval, checkpointing, autoresearch accept/reject, and scaling law fits.

## Scaling Experiments

### Compute Scaling

For each total compute budget C (total training FLOPs):
1. Sweep model size (width, depth) and training steps
2. Constraint: FLOPs_per_step × num_steps ≤ C
3. Tune learning rate
4. Report best energy Wasserstein at each budget

Fit scaling law:
```
energy_wasserstein(C) = a × C^(-α) + floor
```

- **α** (scaling exponent): how fast performance improves with compute. The main result.
- **floor**: irreducible error from finite evaluation samples.

### What to Look For

- How does α change across interaction orders (k=2 vs k=3 vs k=4)? The DiT uses pairwise attention, so three-body and four-body constraints require implicit higher-order learning.
- Temperature interaction: low T (sharp, multi-modal) may show threshold behavior rather than smooth power laws.
- Data scaling: is generation more data-hungry than prediction?
- Mode coverage vs. sample quality: does the model improve on known modes before discovering new ones?

### Secondary Axes

- **Data scaling**: fix model size, vary training set size (10K, 50K, 200K). How data-efficient is the model?
- **Problem scaling**: vary N (10, 20) and T (0.5, 1.0, 2.0). How does difficulty interact with scale?

## Project Structure

```
├── CLAUDE.md
├── configs/
│   ├── train.yaml
│   ├── data/
│   │   └── multibody_{2,23,234}_N{10,20}_T{0.5,1.0,2.0}.yaml
│   ├── model/
│   │   └── transformer.yaml
│   └── logging/
├── data/
│   ├── generate_multibody.py       # Langevin dynamics sampler
│   ├── multibody_dataset.py        # PyTorch dataset
│   └── validate_multibody.py       # Distribution validation plots
├── models/
│   ├── transformer.py              # DiT velocity network
│   └── common.py                   # Shared: timestep embedding, RBF, atom ordering
├── flow_matching/
│   ├── interpolation.py
│   ├── training.py
│   └── sampling.py
├── metrics/
│   ├── energy.py                   # GPU energy computation + Wasserstein metric
│   ├── gr_distance.py              # Pair correlation g(r) utilities
│   └── clash_rate.py               # Pairwise clash detection
├── viz/
│   ├── style.py                    # Global style: fonts, colors, save_figure
│   ├── structure.py                # 3D atom structure plots
│   ├── metrics.py                  # g(r) and min distance histogram
│   ├── scaling.py                  # Scaling curves and capability heatmap
│   └── examples/
│       └── generate_examples.py
├── experiments/
│   ├── train.py                    # Hydra-based training loop
│   ├── evaluate.py                 # Generate samples + compute metrics
│   ├── scaling.py                  # Compute-matched scaling sweep
│   ├── scaling_auto.py             # Fully automated scaling pipeline
│   ├── sweep_hparams.py            # Hyperparameter sweep orchestrator
│   ├── model_registry.py           # Model registry and size presets
│   ├── tasks.py                    # Task abstraction (MultibodyTask)
│   ├── logger.py                   # File-based logging (JSONL)
│   └── checkpointing.py            # Checkpoint management
├── autoresearch/
│   ├── program.md                  # Autoresearch strategy
│   ├── program_session.md          # Session workflow and seeded ideas
│   ├── run.py                      # Quick train+eval harness
│   ├── baseline.py                 # Oracle energy Wasserstein baseline
│   ├── experiment_log.py           # Experiment logging
│   └── visualize.py                # Progress visualization
├── scripts/
│   ├── run_autoresearch.sh
│   ├── run_scaling.sh
│   └── run_sweep.sh
└── tests/
    ├── test_data.py
    ├── test_models.py
    ├── test_flow_matching.py
    └── test_metrics.py
```

## Tech Stack

- PyTorch
- Hydra for configs
- File-based JSONL logging (outputs/logs/)
- numpy for data generation
- Package manager: always use `uv` (never pip)
  - Install packages: `uv add <package>`
  - Run scripts: `uv run <script>`
  - Sync environment: `uv sync`

## Output Directory Convention

All generated artifacts go under `outputs/`, never mixed with source code:

```
outputs/
├── data/{preset}_N{n}_T{temp}/  # Generated .npz datasets
├── plots/                        # All visualizations and figures
├── checkpoints/transformer/      # Model weights
├── logs/transformer/             # Training logs
├── eval/transformer/             # Evaluation results
├── scaling/                      # Scaling law sweep results
└── autoresearch/                 # Autoresearch logs and plots
```

Rules:
- **Never write files to source directories** (`data/`, `metrics/`, `models/`, etc.). All outputs go under `outputs/`.
- **Always use `--output` flags** pointing into `outputs/` when running scripts.
- **Clean up after test/debug runs.** Delete temporary files when done.
- **No stale checkpoints.** Remove superseded or failed experiment checkpoints.
- **Name files descriptively.** Use `{split}.npz` for data, `{description}.png` for plots.
- **The `outputs/` directory is gitignored.** It must never be committed.

## Key Design Decisions

- Single architecture (Transformer/DiT) — scaling behavior is the variable, not architecture comparison
- Flow matching framework shared with potential future architectures
- Same ODE sampler (Euler, same steps) at evaluation
- Random rotation augmentation (model learns rotational invariance from data)
- FLOPs measured with torch profiler — total training FLOPs is the x-axis for scaling curves
- Energy Wasserstein distance as sole evaluation metric — directly measures distributional quality
- All visualization uses `viz/` package with `synthbench_style()` context manager

## Autoresearch Mode

For autonomous algorithm improvement sessions:
- **Start here:** `autoresearch/program.md` — strategy overview
- **Session guide:** `autoresearch/program_session.md` — detailed workflow and seeded ideas

Each session edits `models/transformer.py` and `flow_matching/*.py`. Changes accepted if energy Wasserstein improves.

```bash
# Establish baseline first
uv run python autoresearch/baseline.py --data multibody_23_N20_T1.0

# Run autoresearch
bash scripts/run_autoresearch.sh --data multibody_23_N20_T1.0 --n_gpus 8
```

## Automated Scaling

For fully automated scaling law experiments:
```bash
uv run python experiments/scaling_auto.py --task multibody --n_atoms 20 --archs transformer --n_gpus 8
```
