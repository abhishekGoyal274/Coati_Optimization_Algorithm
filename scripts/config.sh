#!/bin/bash
# =============================================================================
# Experiment Configuration
# =============================================================================

# Benchmark functions to test (F1–F10)
FUNCTIONS=(1 2 3 4 5 6 7 8 9 10)

# Population sizes (number of coatis)
COATIS=(200 400 600 800 1000)

# Iteration counts
ITERS=(500 750 1000)

# OpenMP thread counts
THREADS=(1 2 4 8 16)

# Number of repeated runs per configuration
REPEAT_RUNS=5

# Project root (directory containing the Makefile)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Executable paths
SEQ_EXEC="$PROJECT_ROOT/bin/coa_seq"
OMP_EXEC="$PROJECT_ROOT/bin/coa_omp"
CUDA_EXEC="$PROJECT_ROOT/bin/coa_cuda"

# Log directories
SEQ_LOG_DIR="$PROJECT_ROOT/logs/sequential"
OMP_LOG_DIR="$PROJECT_ROOT/logs/openmp"
CUDA_LOG_DIR="$PROJECT_ROOT/logs/cuda"
