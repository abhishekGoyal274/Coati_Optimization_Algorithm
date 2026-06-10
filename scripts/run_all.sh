#!/bin/bash
# =============================================================================
# Run All Experiments — Build + Sequential + OpenMP
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  COA Full Experiment Pipeline"
echo "============================================"
echo ""

# Step 1: Build
"$SCRIPT_DIR/build.sh"
echo ""

# Step 2: Sequential experiments
echo "============================================"
echo "  Starting Sequential Experiments..."
echo "============================================"
"$SCRIPT_DIR/run_sequential.sh"
echo ""

# Step 3: OpenMP experiments
echo "============================================"
echo "  Starting OpenMP Experiments..."
echo "============================================"
"$SCRIPT_DIR/run_openmp.sh"
echo ""

# Step 3: CUDA experiments (if binary exists)
if [ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bin/coa_cuda" ]; then
    echo "============================================"
    echo "  Starting CUDA Experiments..."
    echo "============================================"
    "$SCRIPT_DIR/run_cuda.sh"
    echo ""
else
    echo "[SKIP] CUDA binary not found — skipping CUDA experiments."
    echo "       Run 'make cuda' to build it."
    echo ""
fi

echo "============================================"
echo "  All experiments completed!"
echo "============================================"
