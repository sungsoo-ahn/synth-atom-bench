# Autoresearch Master Program

## Strategy: Cyclic Per-Architecture Optimization

Optimize one architecture at a time, cycling through all three with a **fixed trial budget per architecture per cycle**. Each architecture gets 24 trials per cycle, then you rotate to the next. When switching architectures, the current flow matching state carries over as a warm start.

```
Cycle 1:
  Transformer   — 24 trials (edit models/transformer.py + flow_matching/*.py)
  Equiv-GNN     — 24 trials (edit models/equiv_gnn.py + flow_matching/*.py, warm start)
  Pairformer    — 24 trials (edit models/pairformer.py + flow_matching/*.py, warm start)

Cycle 2:
  Transformer   — 24 trials (warm start from Pairformer's FM)
  Equiv-GNN     — 24 trials
  Pairformer    — 24 trials

Repeat...
```

### Architecture Order

**Start with Transformer** — it is the most established architecture with the best-understood training dynamics. Improvements found here (especially flow matching tricks like timestep sampling, loss weighting, solver upgrades) are most likely to transfer as warm starts to the other architectures.

Then Equiv-GNN, then Pairformer. This order is fixed across cycles.

### Why 24 Trials per Architecture?

- "Converged" is hard to detect — a string of 3-5 failures doesn't mean the architecture is tapped out, it might mean you're trying the wrong ideas. A hard limit forces you to move on and return later with fresh perspective.
- 24 trials at ~10 min each = ~4 hours per architecture = ~12 hours per full cycle. A 48-hour session completes ~4 full cycles.
- Later cycles benefit from cross-architecture transfer: a flow matching trick that worked for Transformer may have been rejected for GNN, but a GNN architecture change in cycle 1 might make it work in cycle 2.

### Why Per-Architecture Flow Matching?

Different architectures have different inductive biases. A GNN may benefit from velocity prediction while a Transformer does better with x1-prediction. SNR weighting may help attention models more than message-passing networks. Letting each architecture find its own best flow matching recipe produces stronger baselines for the scaling experiments.

## Session Guide

Each session works on **one architecture**. Read `autoresearch/program_session.md` for the full workflow and all seeded ideas.

## GPU Allocation

All 8 GPUs test variants of the current architecture (4 sizes x 2 LRs = 8 jobs). Each variant trains for `--train_time` minutes (default 10). One iteration takes ~10 min wall time.

## Accept/Reject

A change is accepted if the **per-arch aggregate g(r) distance improves** for the target architecture. That's it — no cross-architecture gate.

## Experiment Log

All experiments are logged to `outputs/autoresearch/experiments.jsonl`. Each entry records:
- Target architecture
- Per-variant results
- Whether the change was kept or reverted
- Which files were changed (model, flow matching, or both)

**Always read the log before starting a new iteration.** It's your memory of what has been tried and what worked. Count trials per architecture to know when to rotate.

## Getting Started

```bash
# 1. Establish baseline
uv run python autoresearch/baseline.py --data chain_N50

# 2. Start with Transformer (first architecture in rotation)
#    Read autoresearch/program_session.md, then:
uv run python autoresearch/run.py --archs transformer --data chain_N50 --n_gpus 8 \
  --description "description of the change"

# 3. After 24 trials, rotate to Equiv-GNN
uv run python autoresearch/run.py --archs equiv_gnn --data chain_N50 --n_gpus 8 \
  --description "description of the change"

# 4. After 24 trials, rotate to Pairformer
uv run python autoresearch/run.py --archs pairformer --data chain_N50 --n_gpus 8 \
  --description "description of the change"

# 5. After 24 trials, start Cycle 2 with Transformer again
```

Always pass `--description` so the experiment is auto-logged.

## Key Research References

- Lipman et al., "Flow Matching for Generative Modeling" (2023) — foundation
- Tong et al., "Improving and Generalizing Flow-Based Models with Minibatch OT" (2023) — OT-CFM
- Esser et al., Stable Diffusion 3 (2024) — logit-normal timestep sampling
- "Training Flow Matching: The Role of Weighting and Parameterization" (2025) — SNR weighting, parameterization-architecture interaction
- Karras et al., EDM2 (2024) — EMA, magnitude-preserving design
- Kim et al., AtomMOF (2025) — x1-prediction, L1 loss, time-weighted auxiliary distance loss, SDE sampling
- FlowMol3 (2025) — self-conditioning, late-stage geometry distortion
- SemlaFlow (2024) — scale OT, latent attention
- "Stochastic Sampling from Deterministic Flow Models" (2024) — velocity-to-score SDE conversion
