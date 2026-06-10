#!/bin/bash
# =============================================================================
# Run OpenMP Experiments
# =============================================================================
# Runs the OpenMP COA binary across all parameter combinations defined
# in config.sh, sweeping over thread counts. Each run uses positional
# arguments: <F> <Coatis> <Iters>
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"

# Verify binary exists
if [ ! -f "$OMP_EXEC" ]; then
    echo "ERROR: OpenMP binary not found at $OMP_EXEC"
    echo "       Run ./scripts/build.sh first."
    exit 1
fi

# Ensure log directory exists
mkdir -p "$OMP_LOG_DIR"

total=0
count=0

# Count total experiments
for threads in "${THREADS[@]}"; do
    for func in "${FUNCTIONS[@]}"; do
        for coatis in "${COATIS[@]}"; do
            for iter in "${ITERS[@]}"; do
                for ((run=1; run<=REPEAT_RUNS; run++)); do
                    total=$((total + 1))
                done
            done
        done
    done
done

echo "============================================"
echo "  OpenMP Experiments ($total total runs)"
echo "============================================"
echo ""

for threads in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo "--- Thread count: $threads ---"

    for func in "${FUNCTIONS[@]}"; do
        for coatis in "${COATIS[@]}"; do
            for iter in "${ITERS[@]}"; do
                for ((run=1; run<=REPEAT_RUNS; run++)); do
                    count=$((count + 1))
                    echo "[$count/$total] OMP -> T=$threads F=$func C=$coatis I=$iter RUN=$run"

                    "$OMP_EXEC" "$func" "$coatis" "$iter" "$run"
                done
            done
        done
    done
done

echo ""
echo "============================================"
echo "  OpenMP Experiments — Done!"
echo "  Logs saved to: $OMP_LOG_DIR/"
echo "============================================"
