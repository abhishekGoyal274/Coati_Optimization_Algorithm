# Setup & Run Guide

Step-by-step instructions to build, run, and analyze the Coati Optimization Algorithm HPC project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Building the Project](#building-the-project)
4. [Running Experiments](#running-experiments)
5. [Running the Analysis](#running-the-analysis)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For Sequential + OpenMP (Local Machine / WSL)

| Requirement | Minimum Version | Check Command |
|:---|:---:|:---|
| **g++** | 10+ (C++20 support) | `g++ --version` |
| **OpenMP** | Bundled with g++ | `echo '#include <omp.h>' \| g++ -fopenmp -x c++ -` |
| **make** | Any | `make --version` |
| **Bash** | 4+ | `bash --version` |

**OS**: Linux or WSL (Windows Subsystem for Linux). All shell scripts are written in Bash.

### For CUDA (Google Colab or Local GPU)

| Requirement | Minimum Version | Check Command |
|:---|:---:|:---|
| **nvcc** | 11.0+ (C++17 support) | `nvcc --version` |
| **NVIDIA GPU** | Compute Capability 7.0+ | `nvidia-smi` |
| **CUDA Toolkit** | 11.0+ | `nvcc --version` |

**Recommended**: Google Colab with T4 GPU runtime (free tier).

### For Performance Analysis (Python)

| Requirement | Minimum Version | Install |
|:---|:---:|:---|
| **Python** | 3.8+ | `python3 --version` |
| **NumPy** | 1.20+ | `pip install numpy` |
| **Pandas** | 1.3+ | `pip install pandas` |
| **Matplotlib** | 3.5+ | `pip install matplotlib` |
| **Seaborn** | 0.11+ | `pip install seaborn` |

Install all Python dependencies at once:

```bash
pip install numpy pandas matplotlib seaborn
```

Or with a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib seaborn
```

---

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abhishekGoyal274/Coati_Optimization_Algorithm.git
cd Coati_Optimization_Algorithm
```

### 2. Verify Directory Structure

```bash
ls -la
# You should see: Makefile, README.md, SETUP.md, src/, include/, scripts/
```

### 3. Make Scripts Executable

```bash
chmod +x scripts/*.sh
```

---

## Building the Project

### Build Sequential + OpenMP (Default)

```bash
make
```

This compiles both:
- `bin/coa_seq` — Sequential binary
- `bin/coa_omp` — OpenMP binary

### Build Individually

```bash
make sequential    # Only sequential
make openmp        # Only OpenMP
make cuda          # Only CUDA (requires nvcc)
```

### Build CUDA on Google Colab

```bash
# In a Colab notebook cell:
!make cuda
# Or directly:
!nvcc -O3 -std=c++17 -Iinclude src/coa_cuda_c.cu -o bin/coa_cuda
```

### Clean Build Artifacts

```bash
make clean         # Remove bin/ and logs/ contents
make rebuild       # Clean + rebuild everything
```

Or via the clean script (also removes results/):

```bash
./scripts/clean.sh
```

---

## Running Experiments

### Running a Single Experiment

**Syntax:**
```bash
./bin/<binary> <FunctionID> <NumCoatis> <MaxIterations> <RunNumber>
```

**Examples:**

```bash
# Sequential — F1 (Sphere), 300 coatis, 500 iterations, run #1
./bin/coa_seq 1 300 500 1

# OpenMP — F5 (Rosenbrock), 400 coatis, 1000 iterations, 8 threads, run #1
OMP_NUM_THREADS=8 ./bin/coa_omp 5 400 1000 1

# CUDA — F8 (Schwefel 2.26), 1000 coatis, 1000 iterations, run #1
./bin/coa_cuda 8 1000 1000 1
```

**Parameters:**

| Parameter | Description | Valid Range |
|:---|:---|:---|
| `FunctionID` | Benchmark function index | 1–10 (see README for names) |
| `NumCoatis` | Population size | Any positive integer (tested: 200–1000) |
| `MaxIterations` | Number of optimization iterations | Any positive integer (tested: 500–1000) |
| `RunNumber` | Run identifier (for log filename) | Any positive integer |

**Output**: Log file written to `logs/<implementation>/` directory.

### Running All Experiments (Batch Mode)

The batch scripts iterate over all parameter combinations defined in `scripts/config.sh`:

```bash
# Build + run everything (sequential + OpenMP)
./scripts/run_all.sh

# Or run individually:
./scripts/run_sequential.sh    # ~750 sequential experiments
./scripts/run_openmp.sh        # ~3,750 OpenMP experiments (5 thread counts)
./scripts/run_cuda.sh          # ~750 CUDA experiments (on GPU machine)
```

### Customizing Experiment Parameters

Edit `scripts/config.sh` before running:

```bash
# scripts/config.sh
FUNCTIONS=(1 2 3 4 5 6 7 8 9 10)     # Which benchmark functions
COATIS=(200 400 600 800 1000)         # Population sizes to test
ITERS=(500 750 1000)                  # Iteration counts to test
THREADS=(1 2 4 8 16)                  # OpenMP thread counts
REPEAT_RUNS=5                         # Repeats per config
```

**Quick test** (smaller parameter space):

```bash
# Edit config.sh to:
FUNCTIONS=(1 5)
COATIS=(200 400)
ITERS=(500)
THREADS=(1 4 8)
REPEAT_RUNS=2
```

### Controlling OpenMP Thread Count

```bash
# Set thread count via environment variable
export OMP_NUM_THREADS=8
./bin/coa_omp 1 300 500 1

# Or inline:
OMP_NUM_THREADS=4 ./bin/coa_omp 1 300 500 1
```

---

## Running the Analysis

After experiments complete, generate the analysis report and graphs:

### Run the Python Analysis Script

```bash
python3 scripts/performance_analysis.py
```

This will:
1. Parse all log files from `logs/sequential/`, `logs/openmp/`, and `logs/cuda/`
2. Compute speedup, efficiency, and workload metrics
3. Generate **14 publication-quality PNG graphs**
4. Write `analysis_report.md`, `metrics_summary.csv`, and `metrics_summary.json`
5. Save everything in a timestamped folder under `results/`

### Output Location

```
results/
└── YYYY-MM-DD_HH-MM-SS/
    ├── 01_exec_time_comparison.png
    ├── 02_speedup_by_threads.png
    ├── 03_efficiency_curves.png
    ├── 04_scalability_heatmap.png
    ├── 05_workload_vs_speedup.png
    ├── 06_time_by_population.png
    ├── 07_speedup_distribution.png
    ├── 08_efficiency_heatmap_by_workload.png
    ├── 09_overall_summary_dashboard.png
    ├── 10_convergence_comparison.png
    ├── 11_amdahl_analysis.png
    ├── 12_per_function_time_breakdown.png
    ├── 13_cuda_vs_cpu_speedup.png
    ├── 14_all_implementations_speedup.png
    ├── analysis_report.md
    ├── metrics_summary.csv
    └── metrics_summary.json
```

---

## Troubleshooting

### Build Errors

| Problem | Solution |
|:---|:---|
| `g++: command not found` | Install: `sudo apt install g++` |
| `omp.h: No such file` | Install: `sudo apt install libomp-dev` |
| `nvcc: command not found` | Install CUDA Toolkit or use Google Colab |
| `error: 'clamp' not in 'std'` | Ensure g++ supports C++20: `g++ -std=c++20` |

### Runtime Errors

| Problem | Solution |
|:---|:---|
| `Cannot open log file` | Create dirs first: `mkdir -p logs/sequential logs/openmp logs/cuda` |
| `Permission denied` | Run: `chmod +x scripts/*.sh` |
| Segfault in CUDA | Check: `nvidia-smi` works, GPU has enough memory |
| OpenMP uses 1 thread | Set: `export OMP_NUM_THREADS=8` |

### Analysis Errors

| Problem | Solution |
|:---|:---|
| `ModuleNotFoundError` | Install: `pip install numpy pandas matplotlib seaborn` |
| `No log files found` | Run experiments first (see above) |
| `No sequential data` | Sequential logs are required as baseline for speedup calculation |

### Google Colab Tips

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Build CUDA
!nvcc -O3 -std=c++17 -Iinclude src/coa_cuda_c.cu -o bin/coa_cuda

# Check GPU
!nvidia-smi

# Run CUDA experiment
!./bin/coa_cuda 1 1000 1000 1
```

---

## Typical Workflow

```bash
# 1. Setup
git clone <repo> && cd <repo>
chmod +x scripts/*.sh

# 2. Build
make                    # Sequential + OpenMP
make cuda               # CUDA (if GPU available)

# 3. Run experiments
./scripts/run_all.sh    # Sequential + OpenMP batch
./scripts/run_cuda.sh   # CUDA batch (on GPU machine)

# 4. Analyze
python3 scripts/performance_analysis.py

# 5. View results
ls results/             # Find timestamped folder
cat results/<timestamp>/analysis_report.md
```

---

*For questions or issues, refer to the [README.md](README.md) or the original research paper.*
