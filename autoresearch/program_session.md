# Autoresearch Session Guide

## Overview

You are optimizing one architecture to generate better 3D chain configurations (self-avoiding polymers with fixed bond lengths). You may edit **both** the model and the flow matching code — find the best combination for this architecture.

## Editable Files

For architecture `<arch>` (e.g., `transformer`), you may edit:

- `models/<arch>.py` — the velocity network
- `flow_matching/interpolation.py` — interpolation scheme, noise coupling
- `flow_matching/training.py` — loss computation, timestep sampling, loss weighting
- `flow_matching/sampling.py` — ODE/SDE integration, step scheduling

**Do NOT edit** other model files, `experiments/`, `metrics/`, `data/`, or `autoresearch/`.

## Interface Constraints

These signatures **must be preserved**:

```python
# Model forward
def forward(self, positions: Tensor, t: Tensor) -> Tensor:
    """(batch, N, 3) positions + (batch,) timesteps -> (batch, N, 3) velocities"""

# Flow matching
def interpolate(x_0, t) -> tuple[x_t, noise, velocity_target]
def flow_matching_loss(model, x_0) -> scalar_loss
def sample(model, n_atoms, n_samples, n_steps=100, device="cpu") -> (n_samples, n_atoms, 3)
def sample_batched(model, n_atoms, n_samples, n_steps=100, batch_size=256, device="cpu") -> (n_samples, n_atoms, 3)
```

You may add helper functions, change internals, add parameters with defaults.

## Metric

- **Primary:** `gr_distance` — L1 distance between generated and ground-truth g(r). Lower is better.
- **Secondary (chain-specific):**
  - `bond_violation_rate` — fraction of samples with any bond length deviating > 0.1 from target (1.0)
  - `nonbonded_clash_rate` — fraction of samples with any non-bonded pair closer than 2*radius (0.6)
  - `clash_rate` — fraction with any pairwise overlap (includes bonded pairs, less informative for chains)

## Workflow

1. **Read** `outputs/autoresearch/experiments.jsonl` to see what has been tried
2. **Read** the target model file and `flow_matching/*.py`
3. **Hypothesize** a specific change (model change, FM change, or both)
4. **Implement** the change — one change at a time
5. **Test** (auto-logs when `--description` is provided):
   ```bash
   uv run python autoresearch/run.py --archs <arch> --data chain_N50 --n_gpus 8 \
     --description "brief description of the change"
   ```
   Runs 8 variants in parallel (2 sizes × 4 LRs). Accept/reject is based on **best-of-8** g(r) distance vs previous best.
6. **Parse** JSON output — check `kept`
7. **If `kept` is true** → `git add` the changed files, `git commit -m "autoresearch: [<arch>] <description>"`
   **If not** → `git checkout --` the changed files to revert
8. **Repeat** from step 2 until converged (no improvement for 3-5 iterations)

## Seeded Ideas — Flow Matching

Ordered by impact-to-effort ratio. These apply to all architectures but may help some more than others.

### Tier 1: One-line changes

**FM-1. Logit-normal timestep sampling** (SD3, 2024)
Replace `t ~ Uniform(0,1)` with `t = sigmoid(N(0, 1))`. Concentrates training on intermediate timesteps. In `training.py`:
```python
t = torch.sigmoid(torch.randn(batch_size, device=x_0.device))
```

**FM-2. SNR loss weighting** ("Training Flow Matching", 2025)
Weight the MSE loss by `w(t) = t^2 / (1-t)^2` to emphasize the low-noise regime. In `training.py`:
```python
weight = (t / (1 - t + 1e-5)) ** 2
loss = (weight[:, None, None] * (v_pred - velocity_target) ** 2).mean()
```

**FM-3. L1 loss** (AtomMOF, 2025)
Use L1 instead of L2 — more robust to outliers in geometric data:
```python
return F.l1_loss(v_pred, velocity_target)
```

### Tier 2: Moderate effort

**FM-4. x1-prediction parameterization** (AtomMOF, 2025)
Predict clean data `x_1` instead of velocity. Derive velocity analytically:
```python
x_1_pred = model(x_t, t)
v_derived = (x_1_pred - x_t) / (1 - t[:, None, None] + 1e-5)
loss = F.mse_loss(v_derived, velocity_target)
```
Note: this interacts with architecture. GNNs may prefer velocity; Transformers may prefer x1-prediction.

**FM-5. Heun's method (2nd-order solver)**
In `sampling.py`, replace Euler with predictor-corrector:
```python
v1 = model(x, t_i)
x_pred = x + v1 * dt
v2 = model(x_pred, t_i + dt)
x = x + 0.5 * dt * (v1 + v2)
```

**FM-6. Cosine step spacing**
Concentrate ODE steps near t=1 where trajectories curve most:
```python
t_i = 0.5 * (1 - math.cos(math.pi * i / n_steps))
```

**FM-7. Variance-preserving interpolation**
In `interpolation.py`: `x_t = sqrt(1-t)*noise + sqrt(t)*x_0`. Update velocity target accordingly.

### Tier 3: Higher effort

**FM-8. SDE sampling** (AtomMOF, 2025; "Stochastic Sampling from Deterministic Flow Models", 2024)
Convert ODE to SDE at inference using velocity-to-score conversion:
```python
score = (t * v - x) / ((1 - t) * t + 1e-5)
g2 = alpha * t  # alpha ~ 0.1-0.5
x = x + (v + 0.5 * g2 * score) * dt + sqrt(g2 * dt) * torch.randn_like(x)
```

**FM-9. Time-weighted auxiliary distance loss** (AtomMOF, 2025)
Add pairwise distance loss upweighted at t > 0.5:
```python
t_weight = 1 + 8 * torch.relu(t - 0.5)
dist_loss = t_weight * F.l1_loss(cdist(pred), cdist(true))
loss = main_loss + 0.1 * dist_loss
```

**FM-10. Mini-batch OT coupling**
Use Hungarian algorithm to pair noise with nearest data sample:
```python
from scipy.optimize import linear_sum_assignment
cost = torch.cdist(noise.flatten(1), x_0.flatten(1))
_, col_ind = linear_sum_assignment(cost.cpu().numpy())
noise = noise[col_ind]
```

**FM-11. Predictor-corrector with Langevin**
After each ODE step, run 1-3 score-based corrections:
```python
score = (t * v - x) / ((1 - t) * t + 1e-5)
x = x + step_size * score + sqrt(2 * step_size) * torch.randn_like(x)
```

### Tier 4: Experimental

**FM-12. Self-conditioning** (FlowMol3, 2025)
50% of the time during training, run model twice — feed first prediction as extra input to second pass. Requires adding optional `x_prev` parameter to model forward.

**FM-13. Late-stage geometry distortion** (FlowMol3, 2025)
When t >= 0.5, randomly perturb 20% of atom positions to teach error correction.

## Seeded Ideas — Architecture

### All architectures

**A-1. adaLN-Zero timestep conditioning** (DiT; AtomMOF)
Timestep embedding generates shift/scale/gate per layer. Gates zero-initialized so layers start as identity.

**A-2. Gated output projections** (AtomMOF)
`output = sigmoid(gate_linear(x)) * value_linear(x)` on attention and MLP outputs.

**A-3. Residual scaling**
Scale residual connections by `1/sqrt(num_layers)` or learnable scalar (init 0.1).

**A-4. Pre-norm** (if not already)
LayerNorm before attention/MLP, not after.

**A-5. RMSNorm**
Drop-in replacement for LayerNorm. Removes mean-centering, keeps variance normalization.

**A-6. SiLU activation**
Replace ReLU/GELU with SiLU (Swish). Smooth, non-monotonic, better gradient flow.

### GNN-specific (equiv_gnn.py)

**A-7. Bessel radial basis** (DimeNet): `sqrt(2/c) * sin(k*pi*d/c) / d`. Orthogonal, more expressive per basis function.

**A-8. Envelope cutoff** (DimeNet): Smooth polynomial with zero derivatives at cutoff boundary.

**A-9. Attention-weighted message aggregation**: `alpha_ij = softmax(MLP(edge_ij))`, weighted sum of messages.

**A-10. Vector feature gating**: Gate vector outputs with scalar sigmoid: `v_out = sigmoid(linear(s)) * v_update`.

### Transformer-specific (transformer.py)

**A-11. SwiGLU MLP** (LLaMA): `W2(SiLU(W_gate(x)) * W_up(x))`. Reduce hidden dim by 2/3 to match param count.

**A-12. QK normalization** (ViT-22B): Normalize Q, K before attention with learnable temperature.

**A-13. Continuous pair bias kernel**: Small MLP from distance to per-head attention bias (replaces linear RBF projection).

**A-14. 3D RoPE**: Rotary position embeddings from (x,y,z) coordinates applied to Q, K vectors.

### Pairformer-specific (pairformer.py)

**A-15. Outer product mean** (AF2/AF3): `z_ij += mean(linear(s_i) * linear(s_j))` for single→pair coupling.

**A-16. Alternating triangle direction**: Swap outgoing/incoming order across layers.

**A-17. Richer pair initialization**: Distance bins + relative position MLP + outer product of atom embeddings.

**A-18. Triangle multiplication vs attention**: Try the other variant if only one is implemented.

## Strategy Tips

- **One change at a time.** Even though you can edit both model and FM, change only one thing per iteration. Otherwise you can't attribute the improvement.
- **Start with FM Tier 1** — logit-normal, SNR weighting, L1 loss. These are one-line changes with strong evidence.
- **Then architecture A-1 through A-6** — general improvements that apply to all models.
- **Then arch-specific ideas** — based on which architecture you're working on.
- **Then FM Tier 2-3** — more invasive FM changes once you've exhausted easy wins.
- **Watch per-variant breakdown** — if a change only helps large models, it's about capacity. If it only helps small models, it's better inductive bias.
- **Budget**: Each architecture gets **24 trials per cycle**, then you rotate to the next (see `program.md`). Use all 24 — don't stop early on a streak of failures, try different idea categories instead. You'll return in the next cycle with fresh perspective and warm-started flow matching from the other architectures.
