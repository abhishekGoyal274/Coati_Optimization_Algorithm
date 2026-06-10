# Coati Optimization Algorithm (COA) — HPC Implementation

High Performance Computing implementation of the **Coati Optimization Algorithm (COA)** using:

* Sequential C++
* OpenMP
* CUDA C++ (Work In Progress)

This project benchmarks COA on classical optimization benchmark functions (F1–F11) and analyzes:

* Convergence behavior
* Parallel scalability
* CPU vs GPU performance
* Optimization overheads
* Memory layout effects
* Parallel reduction strategies

Because apparently normal people relax during vacations instead of benchmarking metaheuristics across heterogeneous architectures.

---

## Project Structure

```txt
.
├── Makefile                        # Build system
├── README.md
├── Research Paper.pdf              # Original COA paper
│
├── include/
│   └── benchmark_functions.h      # Benchmark functions F1–F11
│
├── src/
│   ├── coa_sequential.cpp         # Sequential C++ implementation
│   ├── coa_openmp.cpp             # OpenMP implementation
│   └── coa_cuda_c.cu             # CUDA C++ implementation (WIP)
│
├── scripts/
│   ├── config.sh                  # Experiment parameters
│   ├── build.sh                   # Build script
│   ├── clean.sh                   # Clean script
│   ├── run_sequential.sh          # Run sequential experiments
│   ├── run_openmp.sh              # Run OpenMP experiments
│   └── run_all.sh                 # Build + run all experiments
│
├── bin/                            # Compiled binaries (generated)
│   ├── coa_seq
│   └── coa_omp
│
├── logs/                           # Experiment logs (generated)
│   ├── sequential/
│   └── openmp/
│
└── results/                        # Analysis results (future)
```

---

## Quick Start

### Prerequisites

* **g++** with C++20 and OpenMP support
* **Linux / WSL** (scripts are bash)
* **make**

### Build

```bash
# Build both sequential and OpenMP
make

# Or build individually
make sequential
make openmp
```

### Run a Single Experiment

```bash
# Usage: ./bin/<binary> <Function> <Coatis> <Iterations> [RunNumber]

# Sequential — F1, 300 coatis, 500 iterations, run 1
./bin/coa_seq 1 300 500 1

# OpenMP — F5, 400 coatis, 1000 iterations, run 1 (with 8 threads)
OMP_NUM_THREADS=8 ./bin/coa_omp 5 400 1000 1
```

### Run All Experiments (Batch)

```bash
# Build + run everything (sequential + OpenMP)
./scripts/run_all.sh

# Or run individually
./scripts/build.sh            # Build only
./scripts/run_sequential.sh   # Sequential experiments only
./scripts/run_openmp.sh       # OpenMP experiments only
```

### Clean

```bash
# Via Makefile
make clean

# Or via script (also cleans results/)
./scripts/clean.sh
```

---

## Experiment Configuration

Edit `scripts/config.sh` to change experiment parameters:

```bash
FUNCTIONS=(1 2 3 4 5 6 7 8 9 10)     # Benchmark functions
COATIS=(200 400 600 800 1000)         # Population sizes
ITERS=(500 750 1000)                  # Iteration counts
THREADS=(1 2 4 8 16)                  # OpenMP thread counts
REPEAT_RUNS=5                         # Repeats per configuration
```

### Default Parameter Ranges

| Parameter  | Values          |
| ---------- | --------------- |
| Functions  | F1–F10          |
| Coatis     | 200–1000        |
| Iterations | 500–1000        |
| Threads    | 1, 2, 4, 8, 16 |

---

## Implementations

### 1. Sequential C++

* Optimized serial implementation
* Cached fitness values
* Reduced allocations
* Machine-parsable logging

### 2. OpenMP

* Shared-memory parallelism
* Thread-local RNG (thread-safe)
* Reduced synchronization overhead
* Parallel population updates
* Parallel exploration/exploitation phases

#### Important Design Decisions

* Population fitness is monotonically improving
* Current-best == historical-best due to greedy updates

### 3. CUDA C++

Completed high-performance GPU implementation targeting **Google Colab (T4 GPU)**.

#### Implemented Features

*   **One-thread-per-coati design**: Work-sharing across CUDA threads.
*   **GPU fitness evaluation**: Fully executed on device threads.
*   **GPU reduction kernels**: Block-level shared-memory reduction for minimum fitness and index retrieval.
*   **Flattened memory layout**: `double* d_X` mapped to `X[i * DIMENSION + j]` for coalesced memory bandwidth.
*   **Device-side RNG**: Built-in `cuRAND` states initialized and generated directly on the GPU.

#### Build

```bash
# Via Makefile
make cuda

# Or via direct compiler call
nvcc -O3 -std=c++17 -Iinclude src/coa_cuda_c.cu -o bin/coa_cuda
```

---

## Benchmark Functions

| Function | Name          |
| -------- | ------------- |
| F1       | Sphere        |
| F2       | Schwefel 2.22 |
| F3       | Schwefel 1.2  |
| F4       | Schwefel 2.21 |
| F5       | Rosenbrock    |
| F6       | Step Function |
| F7       | Quartic Noise |
| F8       | Schwefel      |
| F9       | Rastrigin     |
| F10      | Ackley        |
| F11      | Griewank      |

---

## Logging Format

All implementations generate machine-parsable logs in `logs/`.

### Example Log

```txt
RUN F=1 COATIS=300 ITER=500 IMPL=SEQ
ITER 0 38168.8
ITER 50 2.29017e-11
ITER 100 4.82757e-27
TIME_MS 412
BEST_SCORE 1.73944e-147
END_RUN
```

### Log File Naming

```txt
logs/sequential/F1_C300_I500.log
logs/openmp/F1_C300_I500.log
```

---

## Optimization Improvements

### Sequential Optimizations

#### Implemented

* Cached fitness values
* Reduced vector allocations
* Improved RNG usage
* Cleaner logging
* Reduced recomputation

#### Planned

* Prefix-sum optimization for F3
* Manual power expansion (`x*x*x*x`)
* Further cache optimizations

### OpenMP Optimizations

#### Implemented

* Parallel region restructuring
* Thread-local random generators
* Reduced critical sections
* Batched synchronization
* Local-best accumulation

---

## CUDA Design Decisions

### Architecture

| Component | Responsibility                       |
| --------- | ------------------------------------ |
| CPU       | Outer loop, logging, timing          |
| GPU       | Population updates, fitness, reduction |

### Memory Layout

```cpp
// CPU/OpenMP: vector<vector<double>>
// CUDA:       double* d_X  →  X[i * DIMENSION + j]
```

Flattened for coalesced memory access and better GPU bandwidth.

### Parallelization

* One CUDA thread per coati
* Each thread updates full solution vector, computes its own fitness
* DIMENSION = 30 is small → minimal synchronization needed

---

## Research Goals

* COA convergence behavior
* Shared-memory parallelization (OpenMP)
* GPU acceleration of metaheuristics (CUDA)
* Synchronization overhead analysis
* Parallel reduction performance
* CPU vs GPU scalability

---

## Future Work

*   CUDA occupancy tuning
*   CUDA memory coalescing analysis
*   Multi-GPU experiments
*   Performance visualization
*   Scalability plots

---

## Notes

*   Logs are intentionally machine-readable for automated plotting.
*   OpenMP uses thread-local RNG for correctness.
*   CUDA version is completed and compile-ready (targeting Google Colab T4 or local GPU).
*   F3 optimization is completed ($O(N^2) \to O(N)$).
*   Sequential and OpenMP run on local machine (WSL/Linux).

---

## Author

HPC project implementing and analyzing the Coati Optimization Algorithm across:

* Sequential C++
* OpenMP
* CUDA C++

Because apparently optimizing artificial raccoons across heterogeneous architectures is a reasonable use of engineering time.
