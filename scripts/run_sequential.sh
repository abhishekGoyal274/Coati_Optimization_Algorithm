#!/bin/bash
# =============================================================================
# Run Sequential Experiments
# =============================================================================
# Runs the sequential COA binary across all parameter combinations defined
# in config.sh. Each run uses positional arguments: <F> <Coatis> <Iters>
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"

# Verify binary exists
if [ ! -f "$SEQ_EXEC" ]; then
    echo "ERROR: Sequential binary not found at $SEQ_EXEC"
    echo "       Run ./scripts/build.sh first."
    exit 1
fi

# Ensure log directory exists
mkdir -p "$SEQ_LOG_DIR"

total=0
count=0

# Count total experiments
for func in "${FUNCTIONS[@]}"; do
    for coatis in "${COATIS[@]}"; do
        for iter in "${ITERS[@]}"; do
            for ((run=1; run<=REPEAT_RUNS; run++)); do
                total=$((total + 1))
            done
        done
    done
done

echo "============================================"
echo "  Sequential Experiments ($total total runs)"
echo "============================================"
echo ""

for func in "${FUNCTIONS[@]}"; do
    for coatis in "${COATIS[@]}"; do
        for iter in "${ITERS[@]}"; do
            for ((run=1; run<=REPEAT_RUNS; run++)); do
                count=$((count + 1))
                echo "[$count/$total] SEQ -> F=$func C=$coatis I=$iter RUN=$run"

                "$SEQ_EXEC" "$func" "$coatis" "$iter" "$run"
            done
        done
    done
done

echo ""
echo "============================================"
echo "  Sequential Experiments — Done!"
echo "  Logs saved to: $SEQ_LOG_DIR/"
echo "============================================"
