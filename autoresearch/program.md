# Autoresearch Master Program

## Goal

Optimize the Transformer architecture and flow matching pipeline for generating multi-body potential chain configurations. Single architecture (Transformer only).

## Current State

### Already Implemented (from previous autoresearch cycles)

**Flow Matching — all accepted:**
- ✅ FM-1: Logit-normal timestep sampling `t = sigmoid(N(0,1))` — in `training.py`
- ✅ FM-2: SNR loss weighting `w(t) = (t/(1-t))^2` — in `training.py`
- ✅ FM-5: Heun's 2nd-order ODE solver — in `sampling.py`
- ✅ FM-6: Cosine step spacing — in `sampling.py`
- ✅ FM-9: Auxiliary distance loss (weight=0.3, t-weighted) — in `training.py`
- ✅ FM-10: Mini-batch OT coupling (greedy approximation) — in `interpolation.py`

**Transformer Architecture — all accepted:**
- ✅ A-1: adaLN-Zero conditioning (zero-init gates) — in `transformer.py`
- ✅ A-3: Residual scaling by `1/sqrt(num_layers)` — in `transformer.py`
- ✅ A-5: RMSNorm on Q, K — in `transformer.py`
- ✅ A-6: SiLU activation — in `transformer.py`
- ✅ A-11: SwiGLU MLP (2/3 hidden dim) — in `transformer.py`
- ✅ A-12: QK normalization — in `transformer.py`
- ✅ A-13: MLP pair bias kernel (RBF → SiLU → per-head) — in `transformer.py`

### Tried but Rejected

**Flow Matching:**
- ❌ FM-3: L1 loss
- ❌ FM-7: Variance-preserving interpolation
- ❌ FM-8: SDE sampling
- ❌ FM-13: Late-stage geometry distortion

**Architecture:**
- ❌ A-2: Gated output projections (tried as FM idea variant)

### Not Yet Tried

**Flow Matching:**
- FM-4: x1-prediction parameterization
- FM-11: Predictor-corrector with Langevin
- FM-12: Self-conditioning

**Architecture:**
- A-14: 3D RoPE

## Session Guide

Each session edits `models/transformer.py` and `flow_matching/*.py`. Read `autoresearch/program_session.md` for the full workflow and all seeded ideas.

## Metric

**Energy Wasserstein distance** (W₁): Lower is better. Primary metric for accept/reject.

## Interface

```bash
# Establish baseline
uv run python autoresearch/baseline.py --data multibody_2_N10_T1.0

# Run autoresearch
uv run python autoresearch/run.py --archs transformer --data multibody_2_N10_T1.0 --n_gpus 8 \
  --description "description of the change"
```

Default grid: 2 sizes (small, medium) × 4 LRs = 8 variants per iteration, ~10 min each.

## GPU Allocation

All 8 GPUs test variants of the Transformer (2 sizes × 4 LRs = 8 jobs). Each variant trains for `--train_time` minutes (default 10). One iteration takes ~10 min wall time.

## Key Research References

- Lipman et al., "Flow Matching for Generative Modeling" (2023) — foundation
- Tong et al., "Improving and Generalizing Flow-Based Models with Minibatch OT" (2023) — OT-CFM
- Esser et al., Stable Diffusion 3 (2024) — logit-normal timestep sampling
- "Training Flow Matching: The Role of Weighting and Parameterization" (2025) — SNR weighting
- Karras et al., EDM2 (2024) — EMA, magnitude-preserving design
- Kim et al., AtomMOF (2025) — x1-prediction, L1 loss, time-weighted auxiliary distance loss
- FlowMol3 (2025) — self-conditioning, late-stage geometry distortion
- SemlaFlow (2024) — scale OT, latent attention
- "Stochastic Sampling from Deterministic Flow Models" (2024) — velocity-to-score SDE conversion
