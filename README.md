# Coati Optimization Algorithm (COA) — HPC Implementation

[![Language](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-parallel-orange.svg)](https://www.openmp.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#)

High-Performance Computing implementation of the **Coati Optimization Algorithm (COA)** across three parallel architectures:

| Implementation | Paradigm | Target Hardware |
|:---:|:---:|:---:|
| Sequential C++ | Baseline | CPU (single core) |
| OpenMP C++ | Shared-memory | Multi-core CPU |
| CUDA C++ | GPU-accelerated | NVIDIA GPU (T4) |

This project benchmarks COA on **10 classical optimization functions** (F1–F10) across **150 unique configurations** (5 population sizes × 3 iteration counts × 10 functions) with **5 repeated runs each**, producing **5,250 total experiment logs**.

---

## Key Results

| Metric | Value |
|:---|:---:|
| Best OpenMP speedup | **2.81×** (T=8) |
| Optimal thread count | **4–8 threads** |
| CUDA avg speedup | **1.53×** over sequential |
| CUDA max speedup | **4.36×** (F8, large workload) |
| Best CUDA function | Schwefel 2.26 (F8) — **2.25× avg** |
| Total experiments | **5,250 runs** |
| Total configurations | **150** |

### Speedup by Thread Count (OpenMP)

| Threads | Avg Speedup | Avg Efficiency |
|:---:|:---:|:---:|
| 1 | 0.73× | 72.6% |
| 2 | 1.22× | 60.8% |
| 4 | 1.75× | 43.7% |
| 8 | 2.04× | 25.5% |
| 16 | 0.37× | 2.3% |

### CUDA Speedup per Function

| Function | Avg Speedup | Max Speedup |
|:---|:---:|:---:|
| Schwefel 2.26 (F8) | 2.25× | 4.36× |
| Rastrigin (F9) | 2.05× | 4.05× |
| Ackley (F10) | 1.94× | 3.87× |
| Step (F6) | 1.38× | 2.65× |
| Sphere (F1) | 1.28× | 2.94× |

### Key Findings

1. **OpenMP sweet spot is 4–8 threads** — diminishing returns beyond 8, performance degradation at 16 (thread management overhead exceeds parallelism benefit).
2. **CUDA excels on compute-heavy functions** — F8, F9, F10 (which involve transcendental operations like `sin`, `cos`, `exp`) achieve 2–4× speedup.
3. **OpenMP runtime overhead is ~27%** — even at T=1, OpenMP is slower than sequential due to thread management.
4. **Larger workloads scale better** — workloads >500K (coatis × iterations) consistently outperform smaller ones.
5. **Amdahl's Law limits** — high serial fraction across all functions limits maximum theoretical speedup.

---

## Project Structure

```
.
├── Makefile                        # Build system (seq + omp + cuda)
├── README.md                       # This file
├── SETUP.md                        # Setup & run guide
├── report.tex                      # LaTeX report with full findings
├── Research Paper.pdf              # Original COA paper (Dehghani et al.)
│
├── include/
│   └── benchmark_functions.h       # Benchmark functions F1–F11 (CPU + header)
│
├── src/
│   ├── coa_sequential.cpp          # Sequential C++ implementation
│   ├── coa_openmp.cpp              # OpenMP parallel implementation
│   └── coa_cuda_c.cu              # CUDA GPU implementation
│
├── scripts/
│   ├── config.sh                   # Experiment parameters
│   ├── build.sh                    # Build script
│   ├── clean.sh                    # Clean script
│   ├── run_sequential.sh           # Run sequential experiments
│   ├── run_openmp.sh               # Run OpenMP experiments
│   ├── run_cuda.sh                 # Run CUDA experiments
│   ├── run_all.sh                  # Build + run all experiments
│   ├── analyze_perf.sh             # Trigger performance analysis
│   └── performance_analysis.py     # Python analysis + graph generation
│
├── bin/                            # Compiled binaries (generated)
│   ├── coa_seq
│   ├── coa_omp
│   └── coa_cuda
│
├── logs/                           # Experiment logs (generated)
│   ├── sequential/                 # F{n}_C{pop}_I{iter}_R{run}.log
│   ├── openmp/                     # F{n}_C{pop}_I{iter}_T{threads}_R{run}.log
│   └── cuda/                       # F{n}_C{pop}_I{iter}_R{run}.log
│
└── results/                        # Analysis output (generated)
    └── YYYY-MM-DD_HH-MM-SS/       # Timestamped run
        ├── 01–14 PNG graphs        # Publication-quality visualizations
        ├── analysis_report.md      # Auto-generated analysis report
        ├── metrics_summary.csv     # Raw metrics data
        └── metrics_summary.json    # Machine-readable summary
```

---

## Implementations

### 1. Sequential C++ (`coa_sequential.cpp`)

Optimized serial baseline implementation:
- Cached fitness values to avoid recomputation
- Pre-allocated scratch vectors (no per-iteration allocation)
- Mersenne Twister RNG (`mt19937_64`) for high-quality randomness
- Machine-parsable log output for automated analysis

### 2. OpenMP (`coa_openmp.cpp`)

Shared-memory parallel implementation:
- **Thread-local RNG**: Each thread has its own `mt19937_64` engine seeded with `random_device + thread_id`, ensuring correctness and no lock contention
- **Parallel population initialization** and fitness evaluation
- **Parallel exploitative + exploration phases** with `#pragma omp for`
- **Minimized critical sections**: Local-best accumulation pattern reduces synchronization to once per initialization
- **`#pragma omp single`** for serialized logging and best-index updates
- **`nowait` clauses** where safe to reduce implicit barriers

### 3. CUDA C++ (`coa_cuda_c.cu`)

GPU-accelerated implementation targeting **Google Colab T4**:
- **One-thread-per-coati** design — each CUDA thread manages a full solution vector
- **Device-side fitness evaluation** — all 10 benchmark functions re-implemented as `__device__` functions
- **cuRAND on-device RNG** — `curandState` per thread, initialized once, reused across iterations
- **Block-level shared-memory reduction** — custom kernel finds global best with `O(log n)` parallel reduction
- **Flattened memory layout** — `double* d_X` indexed as `X[i * DIMENSION + j]` for coalesced GPU memory access
- **Separate kernels** for exploitative phase, local bounds computation, and exploration phase

---

## Benchmark Functions

| Function | Name | Search Range | Global Minimum |
|:---:|:---|:---:|:---:|
| F1 | Sphere | [-100, 100] | 0 |
| F2 | Schwefel 2.22 | [-10, 10] | 0 |
| F3 | Schwefel 1.2 | [-100, 100] | 0 |
| F4 | Schwefel 2.21 | [-100, 100] | 0 |
| F5 | Rosenbrock | [-30, 30] | 0 |
| F6 | Step Function | [-100, 100] | 0 |
| F7 | Quartic + Noise | [-1.28, 1.28] | 0 |
| F8 | Schwefel 2.26 | [-500, 500] | -12569.5 |
| F9 | Rastrigin | [-5.12, 5.12] | 0 |
| F10 | Ackley | [-32, 32] | 0 |

All functions use **DIMENSION = 30**.

---

## Analysis & Visualization

The project generates **14 publication-quality graphs** via `scripts/performance_analysis.py`:

| # | Graph | What It Shows |
|:---:|:---|:---|
| 01 | Execution Time Comparison | Seq vs OMP vs CUDA bar chart per function |
| 02 | Speedup by Threads | Speedup curves across T=1–16 per function |
| 03 | Efficiency Curves | Parallel efficiency vs thread count |
| 04 | Scalability Heatmap | Speedup heatmap: functions × threads |
| 05 | Workload vs Speedup | How workload size affects parallel gain |
| 06 | Time by Population | Execution time vs population size |
| 07 | Speedup Distribution | Violin plot of speedup by thread count |
| 08 | Efficiency Heatmap | Efficiency across workload × threads |
| 09 | Summary Dashboard | 6-panel overview of key metrics |
| 10 | Convergence Comparison | Convergence curves: Seq vs OMP vs CUDA |
| 11 | Amdahl's Analysis | Actual speedup vs Amdahl's Law predictions |
| 12 | Per-Function Time | Grouped bar chart across implementations |
| 13 | CUDA vs CPU Speedup | GPU speedup analysis per function |
| 14 | All Implementations | Combined speedup comparison |

---

## Experiment Configuration

Edit `scripts/config.sh` to modify parameters:

```bash
FUNCTIONS=(1 2 3 4 5 6 7 8 9 10)     # Benchmark functions
COATIS=(200 400 600 800 1000)         # Population sizes
ITERS=(500 750 1000)                  # Iteration counts
THREADS=(1 2 4 8 16)                  # OpenMP thread counts
REPEAT_RUNS=5                         # Repeats per configuration
```

---

## Logging Format

All implementations produce machine-parsable logs:

```
RUN F=1 COATIS=300 ITER=500 IMPL=SEQ
ITER 0 38168.8
ITER 50 2.29017e-11
ITER 100 4.82757e-27
TIME_MS 412
BEST_SCORE 1.73944e-147
END_RUN
```

Log file naming:
```
logs/sequential/F1_C300_I500_R1.log
logs/openmp/F1_C300_I500_T8_R1.log
logs/cuda/F1_C300_I500_R1.log
```

---

## Quick Start

> For detailed setup instructions, see [SETUP.md](SETUP.md).

```bash
# Build sequential + OpenMP
make

# Run a single experiment
./bin/coa_seq 1 300 500 1
OMP_NUM_THREADS=8 ./bin/coa_omp 5 400 1000 1

# Run all experiments
./scripts/run_all.sh

# Generate analysis
python3 scripts/performance_analysis.py
```

---

## References

- Dehghani, M., Montazeri, Z., Trojovská, E., & Trojovský, P. (2023). *Coati Optimization Algorithm: A new bio-inspired metaheuristic algorithm for solving optimization problems.* Knowledge-Based Systems, 259, 110011.

---

## Author

**Abhishek Goyal**
Summer Research Project, SVNIT (Sardar Vallabhbhai National Institute of Technology)

---

*An HPC project implementing and analyzing the Coati Optimization Algorithm across sequential C++, OpenMP, and CUDA C++ — because apparently optimizing artificial coatis across heterogeneous architectures is a reasonable use of engineering time.* 🦝
