#!/bin/bash
# =============================================================================
# Build Script — compiles sequential + openmp targets
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  Building COA — Sequential + OpenMP"
echo "============================================"
echo ""

echo "[1/2] Building Sequential..."
make sequential
echo ""

echo "[2/2] Building OpenMP..."
make openmp
echo ""

echo "============================================"
echo "  Build complete!"
echo "============================================"
