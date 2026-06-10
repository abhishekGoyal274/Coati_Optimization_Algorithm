#!/bin/bash
# =============================================================================
# Clean Script — removes binaries, logs, and results
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Cleaning binaries..."
rm -rf bin/*

echo "Cleaning logs..."
rm -rf logs/*

echo "Cleaning results..."
rm -rf results/*

echo "[OK] All cleaned."
