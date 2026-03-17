#!/bin/bash
# Autonomous autoresearch loop. Restarts Claude Code on exit (context exhaustion, crash, etc.)
# Usage: bash scripts/run_autoresearch.sh [--hours 48] [--n_gpus 8] [--data chain_N50]

set -euo pipefail

HOURS=48
N_GPUS=8
DATA="multibody_23_N50_T1.0"
# Max time per Claude invocation (seconds). If Claude hangs, kill and restart.
# 2 hours is generous — a single iteration is ~10 min, so Claude should complete
# many iterations well within this. Context exhaustion normally triggers exit first.
CLAUDE_TIMEOUT=7200

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours)           HOURS="$2"; shift 2 ;;
        --n_gpus)          N_GPUS="$2"; shift 2 ;;
        --data)            DATA="$2"; shift 2 ;;
        --claude_timeout)  CLAUDE_TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

END=$(($(date +%s) + HOURS * 3600))
ITERATION=0

echo "=== Autoresearch session ==="
echo "Duration: ${HOURS}h | GPUs: ${N_GPUS} | Data: ${DATA}"
echo "Claude timeout: ${CLAUDE_TIMEOUT}s per invocation"
echo "End time: $(date -d @${END} 2>/dev/null || date -r ${END})"
echo ""

# Ensure baseline exists (don't let set -e kill the loop if data is missing)
if [ ! -f "outputs/autoresearch/baselines.json" ]; then
    echo "Computing oracle baseline..."
    if ! uv run python autoresearch/baseline.py --data "$DATA"; then
        echo "WARNING: baseline computation failed. Continuing anyway — harness will run without baseline."
    fi
    echo ""
fi

while [ "$(date +%s)" -lt "$END" ]; do
    ITERATION=$((ITERATION + 1))
    REMAINING=$(( (END - $(date +%s)) / 60 ))
    echo "=== Restart #${ITERATION} | ${REMAINING} min remaining ==="

    # Run Claude in its own process group (setsid) so we can kill all children
    # (training subprocesses) when the timeout fires, not just the parent.
    PROMPT="$(cat <<EOF
You are running an autonomous autoresearch session. Read autoresearch/program.md for the strategy and autoresearch/program_session.md for the detailed workflow.

Setup:
- Data: ${DATA}
- GPUs: ${N_GPUS}
- Time remaining in session: ${REMAINING} minutes

Architecture: transformer only. Up to 48 trials or until no improvement.

Instructions:
1. Read outputs/autoresearch/experiments.jsonl to see what has been tried.
2. Count trials in the log.
3. For each iteration:
   a. Read autoresearch/program_session.md for seeded ideas.
   b. Read the target model file (models/<arch>.py) and flow_matching/*.py.
   c. Hypothesize and implement ONE change (model, flow matching, or both).
   d. Run the harness with --description for auto-logging:
      uv run python autoresearch/run.py --archs <arch> --data ${DATA} --n_gpus ${N_GPUS} --description "[<arch>] <brief description>"
   e. Parse the JSON output. Check the "kept" field.
   f. If kept is true: git add the changed files, git commit with "autoresearch: [<arch>] <description>".
   g. If kept is false: git checkout the changed files to revert.
   The --description flag auto-logs to experiments.jsonl — no separate logging step needed.
4. Repeat step 3 until converged or 48 trials.
EOF
    )"

    setsid bash -c "claude --dangerously-skip-permissions -p \"\$1\"" _ "$PROMPT" &
    CLAUDE_PID=$!

    # Wait up to CLAUDE_TIMEOUT for the process group to finish
    if ! timeout "${CLAUDE_TIMEOUT}" tail --pid=$CLAUDE_PID -f /dev/null 2>/dev/null; then
        echo "Claude TIMED OUT after ${CLAUDE_TIMEOUT}s at $(date). Killing process group..."
        # Kill the entire process group (Claude + all training subprocesses)
        kill -- -"$CLAUDE_PID" 2>/dev/null || true
        sleep 5
        # Force-kill any survivors
        kill -9 -- -"$CLAUDE_PID" 2>/dev/null || true
        # Also kill any quick_train_eval subprocesses that escaped the group
        pkill -f "quick_train_eval" 2>/dev/null || true
        # Revert any uncommitted changes to prevent corrupted state
        git checkout -- flow_matching/ models/ 2>/dev/null || true
        sleep 5
    else
        wait $CLAUDE_PID && EXIT_CODE=0 || EXIT_CODE=$?
        echo "Claude exited (code=${EXIT_CODE:-0}) at $(date). Restarting in 5s..."
        sleep 5
    fi
done

echo ""
echo "=== Session complete after ${HOURS}h ==="
echo "Check outputs/autoresearch/experiments.jsonl for results."
echo "Run: uv run python autoresearch/visualize.py --log outputs/autoresearch/experiments.jsonl --output outputs/autoresearch/plots"
