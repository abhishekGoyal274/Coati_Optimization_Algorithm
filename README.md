# Coati Optimization Algorithm (COA) - HPC Implementation

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

# Project Structure

```txt
.
├── _main.cpp                  # Sequential C++ implementation
├── _main_openmp.cpp           # OpenMP implementation
├── _main_cuda_c.cu            # CUDA C++ implementation (WIP)
├── _benchmark_functions.h     # Benchmark functions F1–F11
│
├── _main_logs/                # Sequential logs
├── _main_openmp_logs/         # OpenMP logs
├── _main_cuda_c_logs/         # CUDA logs
│
├── run_seq.ps1                # Sequential batch script
├── run_openmp.ps1             # OpenMP batch script
├── run_cuda.ps1               # CUDA batch script
│
└── README.md
```

---

# Implementations

## 1. Sequential C++

### Features

* Optimized serial implementation
* Cached fitness values
* Reduced allocations
* Machine-parsable logging
* Benchmark automation

### Compile

```bash
g++ -O3 _main.cpp -o _main.exe
```

### Run

```bash
./_main.exe 1 300 500
```

### Arguments

```txt
./_main.exe <Function> <Coatis> <Iterations>
```

### Example

```bash
./_main.exe 5 400 1000
```

---

## 2. OpenMP Version

### Features

* Shared-memory parallelism
* Thread-local RNG
* Reduced synchronization overhead
* Parallel population updates
* Parallel exploration/exploitation phases

### Compile

```bash
g++ -O3 -fopenmp _main_openmp.cpp -o _main_openmp.exe
```

### Run

```bash
./_main_openmp.exe 1 300 500
```

### Set Thread Count

#### Linux/macOS

```bash
export OMP_NUM_THREADS=8
```

#### Windows PowerShell

```powershell
$env:OMP_NUM_THREADS=8
```

---

## 3. CUDA C++ (WIP)

### Planned Features

* One-thread-per-coati design
* GPU fitness evaluation
* GPU reduction kernels
* Flattened memory layout
* Device-side RNG
* CUDA parallel benchmark execution

### Planned Compile

```bash
nvcc -O3 _main_cuda_c.cu -o _main_cuda_c.exe
```

---

# Benchmark Functions

Implemented benchmark functions:

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

# Optimization Improvements

## Sequential Optimizations

### Implemented

* Cached fitness values
* Reduced vector allocations
* Improved RNG usage
* Cleaner logging
* Reduced recomputation

### Planned

* Prefix-sum optimization for F3
* Manual power expansion (`x*x*x*x`)
* Further cache optimizations

---

## OpenMP Optimizations

### Implemented

* Parallel region restructuring
* Thread-local random generators
* Reduced critical sections
* Batched synchronization
* Local-best accumulation

### Important Design Decisions

* Population fitness is monotonically improving
* Current-best == historical-best due to greedy updates

---

# Logging Format

All implementations generate machine-parsable logs.

### Example

```txt
RUN F=1 COATIS=300 ITER=500 IMPL=OpenMP
ITER 0 38168.8
ITER 50 2.29017e-11
ITER 100 4.82757e-27
TIME_MS 412
BEST_SCORE 1.73944e-147
END_RUN
```

---

# Batch Execution

PowerShell scripts automate experiments across:

* Multiple benchmark functions
* Multiple coati counts
* Multiple iteration counts

### Example Parameter Ranges

```powershell
$functions  = 1..10
$coatisList = @(200, 400, 600, 800, 1000)
$itersList  = @(500, 750, 1000)
```

---

# CUDA Design Decisions

The CUDA implementation is being designed carefully before coding to avoid architectural rewrites later.

## Selected Architecture

### CPU Responsibilities

* Outer iteration loop
* Logging
* Timing
* Experiment orchestration

### GPU Responsibilities

* Population updates
* Fitness evaluation
* Reduction kernels

---

## Memory Layout

### CPU/OpenMP

```cpp
vector<vector<double>>
```

### CUDA

```cpp
double* d_X
```

### Flattened Indexing

```cpp
X[i * DIMENSION + j]
```

### Reason

* Coalesced memory access
* Better GPU bandwidth utilization
* Contiguous layout

---

## Parallelization Strategy

### Selected

* One CUDA thread per coati

Each thread:

* Updates full solution vector
* Computes its own fitness
* Stores fitness independently

### Reason

* DIMENSION = 30 is small
* Minimal synchronization
* Simple reduction design

---

# Research Goals

This project studies:

* COA convergence behavior
* Shared-memory parallelization
* GPU acceleration of metaheuristics
* Synchronization overhead
* Parallel reduction performance
* CPU vs GPU scalability

---

# Future Work

Planned improvements:

* Complete CUDA implementation
* Shared-memory CUDA reductions
* cuRAND / XORShift RNG
* CUDA occupancy tuning
* CUDA memory coalescing analysis
* Multi-GPU experiments
* Performance visualization
* Scalability plots

---

# Example Commands

## Sequential

```bash
./_main.exe 1 300 500
```

## OpenMP

```bash
./_main_openmp.exe 1 300 500
```

## CUDA (Planned)

```bash
./_main_cuda_c.exe 1 300 500
```

---

# Experimental Parameters

Typical experiment ranges:

| Parameter  | Values   |
| ---------- | -------- |
| Functions  | F1–F10   |
| Coatis     | 200–1000 |
| Iterations | 500–1000 |

---

# Notes

* Logs are intentionally machine-readable for automated plotting.
* OpenMP uses thread-local RNG for correctness.
* CUDA version is currently under development.
* Benchmark functions are currently executed serially.
* F3 optimization pending (O(n²) → O(n)).

---

# Author

HPC project implementing and analyzing the Coati Optimization Algorithm across:

* Sequential C++
* OpenMP
* CUDA C++

Because apparently optimizing artificial raccoons across heterogeneous architectures is a reasonable use of engineering time.
