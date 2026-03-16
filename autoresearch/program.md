# Autoresearch Master Program

## Two-Phase Improvement Cycle

Each autoresearch session alternates between two phases. This structure exists because **flow matching changes depend on all architectures** (they're shared infrastructure), while **architecture changes are independent** (each model file is self-contained).

### Phase 1: Architecture Improvements

Improve each architecture independently. Since the three model files don't interact, you can work on them in any order (or even in parallel across separate sessions).

- **Editable:** `models/{arch}.py` (one at a time)
- **Frozen:** `flow_matching/`, `experiments/`, everything else
- **Test command:** `uv run python autoresearch/run.py --archs <arch> --data hard_sphere_N50 --n_gpus 8`
- **Accept criterion:** aggregate g(r) distance improves for that single architecture
- **Instructions:** See `autoresearch/program_arch.md`

**Why architecture first?** Flow matching changes are tested against all three architectures. If you improve the architectures first, then the flow matching tests run on stronger models — this gives flow matching changes a fairer evaluation and lets improvements compound.

### Phase 2: Flow Matching Improvements

Improve the shared flow matching framework. Since this code is used by all architectures, changes must help across the board.

- **Editable:** `flow_matching/*.py`
- **Frozen:** `models/`, `experiments/`, everything else
- **Test command:** `uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --n_gpus 8`
- **Accept criterion:** `all_archs_improved` is true — every architecture must benefit
- **Instructions:** See `autoresearch/program_flow.md`

**Why test all architectures?** A flow matching change that helps Transformers but hurts GNNs isn't a framework improvement — it's an architecture-specific interaction. Requiring all-arch improvement ensures we only keep changes that are genuinely better for the framework.

## Session Structure

A typical session runs one or more cycles:

```
Cycle N:
  Phase 1 — Architecture
    1a. Pick an architecture (e.g. transformer)
    1b. Read program_arch.md + experiment log
    1c. Iterate: hypothesize → implement → test → accept/revert
    1d. Repeat 1c for 3-5 iterations
    1e. Optionally switch to another architecture and repeat 1a-1d

  Phase 2 — Flow Matching
    2a. Read program_flow.md + experiment log
    2b. Iterate: hypothesize → implement → test (--archs all) → accept/revert
    2c. Repeat 2b for 3-5 iterations

  Repeat cycle
```

You don't need to be rigid about this — if you have a strong flow matching hypothesis early, go straight to Phase 2. The phases are a guide, not a rule.

## GPU Allocation

With 8 GPUs available:

- **Architecture phase:** All 8 GPUs test variants of one architecture (4 sizes × 2 LRs). Each variant runs for `--train_time` minutes (default 10), so one iteration takes ~10 min wall time.

- **Flow matching phase:** All 8 GPUs test variants of one architecture at a time. The harness runs each architecture sequentially (equiv_gnn → transformer → pairformer), so one iteration takes ~30 min wall time (3 × 10 min).

## Accept/Reject Rules

| Phase | Accept if... | Reject if... |
|-------|-------------|-------------|
| Architecture | Aggregate g(r) improves for that arch | Any degradation |
| Flow matching | `all_archs_improved` is true | Any arch degrades |

## Experiment Log

Both phases share the same log at `outputs/autoresearch/experiments.jsonl`. Each entry records:
- Which phase (arch or flow matching)
- Which arch(es) were tested
- Per-arch and per-variant results
- Whether the change was kept or reverted

**Always read the log before starting a new iteration.** It's your memory of what has been tried and what worked.

## Getting Started

```bash
# 1. Establish baseline
uv run python autoresearch/baseline.py --data hard_sphere_N50

# 2. Start with architecture improvements
#    Read autoresearch/program_arch.md, then:
uv run python autoresearch/run.py --archs equiv_gnn --data hard_sphere_N50 --n_gpus 8

# 3. Switch to flow matching improvements
#    Read autoresearch/program_flow.md, then:
uv run python autoresearch/run.py --archs all --data hard_sphere_N50 --n_gpus 8
```
