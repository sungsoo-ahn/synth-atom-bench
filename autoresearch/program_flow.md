# Autoresearch Program: Flow Matching Improvements

## Overview

You are autonomously improving the flow matching framework to generate better 3D hard sphere configurations. You will iteratively edit flow matching code, test changes, and keep what works.

**A flow matching change affects all architectures.** The harness tests all three in one invocation and only accepts changes that improve across the board.

## Editable Files

You may **only** edit these files:
- `flow_matching/interpolation.py` — interpolation scheme (linear, VP, etc.)
- `flow_matching/training.py` — loss computation (weighting, noise schedule, etc.)
- `flow_matching/sampling.py` — ODE integration (step schedule, solver, etc.)

## Frozen Files (DO NOT EDIT)

Everything else is frozen: `experiments/`, `metrics/`, `data/`, `autoresearch/`, `models/`.

## Interface Constraints

These function signatures **must be preserved** — the harness calls them:

```python
# flow_matching/interpolation.py
def interpolate(x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (x_t, noise, velocity_target)"""

# flow_matching/training.py
def flow_matching_loss(model: nn.Module, x_0: Tensor) -> Tensor:
    """Returns scalar loss"""

# flow_matching/sampling.py
def sample(model, n_atoms, n_samples, n_steps=100, device="cpu") -> Tensor:
    """Returns (n_samples, n_atoms, 3)"""

def sample_batched(model, n_atoms, n_samples, n_steps=100, batch_size=256, device="cpu") -> Tensor:
    """Returns (n_samples, n_atoms, 3)"""
```

You may add helper functions, change internals, add parameters with defaults — just preserve the existing call signatures.

## Metric

- **Primary:** `gr_distance` (L1 distance between generated and ground-truth pair correlation function). Lower is better.
- **Secondary:** `clash_rate` (fraction of samples with atomic overlaps). Reported but not used for accept/reject.
- **Baseline:** Run `uv run python autoresearch/baseline.py --data hard_sphere_N50` to see the oracle floor.

## Running the Harness

```bash
# Test across ALL architectures in one invocation (required for flow matching changes)
uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --n_gpus 8

# Shorter runs for quick iteration
uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --train_time 5 --n_gpus 8

# Single-GPU testing during development (slow — runs 8 variants × 3 archs sequentially)
uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --train_time 5 --n_gpus 1
```

Each variant trains for `--train_time` minutes (default 10). With `--archs all`, the harness runs the variant grid (4 sizes × 2 LRs = 8 jobs) for each of the 3 architectures and reports:

- **Per-arch results**: variant breakdown and per-arch aggregate g(r) distance
- **`aggregate_metric`**: grand mean across all archs and variants
- **`all_archs_improved`**: True only if every architecture improved — this is the accept criterion

## Workflow

1. **Read this file** and `outputs/autoresearch/experiments.jsonl` (history of past attempts)
2. **Read current** `flow_matching/*.py` files
3. **Hypothesize** a specific change with a clear rationale
4. **Implement** the change
5. **Test across all architectures**:
   ```bash
   uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --n_gpus 8
   ```
6. **Parse** the JSON output — check `all_archs_improved`
7. **If `all_archs_improved` is true** → `git add flow_matching/ && git commit -m "autoresearch: <description>"`
   **If not** → `git checkout -- flow_matching/`
8. **Log the result** using `autoresearch/experiment_log.py`:
   ```python
   from autoresearch.experiment_log import log_experiment
   log_experiment("outputs/autoresearch/experiments.jsonl", ...)
   ```
9. **Repeat** from step 2

## Seeded Ideas

Try these in roughly this order (easiest wins first):

1. **Timestep importance sampling** — Sample t from a non-uniform distribution (e.g., Beta or logit-normal) to focus training on harder timesteps near t=0 and t=1.

2. **Loss weighting by timestep** — Weight the MSE loss by a function of t (e.g., `w(t) = 1/(1-t+eps)` to emphasize near-data timesteps).

3. **Variance-preserving interpolation** — Replace linear `x_t = (1-t)*noise + t*x_0` with `x_t = sqrt(1-t)*noise + sqrt(t)*x_0` (preserves variance at all t).

4. **Noise schedule** — Use a non-linear mapping `s(t)` before interpolation (e.g., cosine schedule from DDPM).

5. **Velocity parameterization** — Instead of predicting `x_0 - noise`, predict `x_0` directly and derive velocity, or predict noise and derive velocity.

6. **Adaptive ODE steps** — Use more steps near t=0 (data end) where trajectories curve most. E.g., cosine step spacing.

7. **Higher-order ODE solver** — Replace Euler with Heun's method (2nd order) or RK4 in the sampler.

8. **Stochastic sampling** — Add small noise at each ODE step (SDE formulation) to improve sample diversity.

## Tips

- Make **one change at a time**. Multiple changes make it impossible to attribute improvements.
- Keep changes **small and reversible**. Prefer adding a parameter with a default over restructuring.
- **A good flow matching change helps all architectures.** If it helps 2/3 but hurts 1, reject it — it's not a true framework improvement, it's an architecture-specific interaction.
- Watch for changes that only help at one learning rate or model size — robust improvements help across the grid.
- The experiment log is your memory. Read it before each iteration to avoid repeating failed ideas.
