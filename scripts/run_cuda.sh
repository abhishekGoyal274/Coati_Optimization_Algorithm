#!/bin/bash
# =============================================================================
# Run CUDA Experiments
# =============================================================================
# Runs the CUDA COA binary across all parameter combinations defined
# in config.sh. Each run uses positional arguments: <F> <Coatis> <Iters>
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"

# Verify binary exists
if [ ! -f "$CUDA_EXEC" ]; then
    echo "ERROR: CUDA binary not found at $CUDA_EXEC"
    echo "       Run 'make cuda' first."
    exit 1
fi

# Ensure log directory exists
mkdir -p "$CUDA_LOG_DIR"

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
echo "  CUDA Experiments ($total total runs)"
echo "============================================"
echo ""

for func in "${FUNCTIONS[@]}"; do
    for coatis in "${COATIS[@]}"; do
        for iter in "${ITERS[@]}"; do
            for ((run=1; run<=REPEAT_RUNS; run++)); do
                count=$((count + 1))
                echo "[$count/$total] CUDA -> F=$func C=$coatis I=$iter RUN=$run"

                "$CUDA_EXEC" "$func" "$coatis" "$iter" "$run"
            done
        done
    done
done

echo ""
echo "============================================"
echo "  CUDA Experiments — Done!"
echo "  Logs saved to: $CUDA_LOG_DIR/"
echo "============================================"
