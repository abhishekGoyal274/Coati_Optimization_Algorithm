#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/benchmark_functions.h"

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while (0)

struct Element {
    double val;
    int idx;
};

/* ---------------- CUDA Device Benchmark Functions ---------------- */

__device__ double gpu_F1(const double* x, int dim) {
    double R = 0.0;
    for (int i = 0; i < dim; ++i) R += x[i] * x[i];
    return R;
}

__device__ double gpu_F2(const double* x, int dim) {
    double s = 0.0, p = 1.0;
    for (int i = 0; i < dim; ++i) {
        double a = fabs(x[i]);
        s += a;
        p *= a;
    }
    return s + p;
}

__device__ double gpu_F3(const double* x, int dim) {
    double R = 0.0;
    double running_sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        running_sum += x[i];
        R += running_sum * running_sum;
    }
    return R;
}

__device__ double gpu_F4(const double* x, int dim) {
    double mx = 0.0;
    for (int i = 0; i < dim; ++i) {
        double a = fabs(x[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

__device__ double gpu_F5(const double* x, int dim) {
    double R = 0.0;
    for (int i = 0; i < dim - 1; ++i) {
        double t1 = x[i+1] - x[i]*x[i];
        double t2 = x[i] - 1.0;
        R += 100.0 * t1 * t1 + t2 * t2;
    }
    return R;
}

__device__ double gpu_F6(const double* x, int dim) {
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        double t = floor(x[i] + 0.5);
        R += t * t;
    }
    return R;
}

__device__ double gpu_F7(const double* x, int dim, curandState* state) {
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        double x2 = x[i] * x[i];
        R += (i + 1) * x2 * x2;
    }
    R += curand_uniform_double(state);
    return R;
}

__device__ double gpu_F8(const double* x, int dim) {
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        R += -x[i] * sin(sqrt(fabs(x[i])));
    }
    return R;
}

__device__ double gpu_F9(const double* x, int dim) {
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        R += x[i]*x[i] - 10.0 * cos(2.0 * 3.14159265358979323846 * x[i]);
    }
    return R + 10.0 * dim;
}

__device__ double gpu_F10(const double* x, int dim) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum1 += x[i]*x[i];
        sum2 += cos(2.0 * 3.14159265358979323846 * x[i]);
    }
    return -20.0 * exp(-0.2 * sqrt(sum1 / dim))
           - exp(sum2 / dim) + 20.0 + 2.71828182845904523536;
}

__device__ double gpu_F11(const double* x, int dim) {
    double sum2 = 0.0, prod = 1.0;
    for (int i = 0; i < dim; ++i) {
        double v = x[i];
        sum2 += v*v / 4000.0;
        prod *= cos(v / sqrt(double(i+1)));
    }
    return sum2 - prod + 1.0;
}

__device__ double gpu_evaluation_fitness(const double* x, int F, int dim, curandState* state) {
    switch (F) {
        case 1:  return gpu_F1(x, dim);
        case 2:  return gpu_F2(x, dim);
        case 3:  return gpu_F3(x, dim);
        case 4:  return gpu_F4(x, dim);
        case 5:  return gpu_F5(x, dim);
        case 6:  return gpu_F6(x, dim);
        case 7:  return gpu_F7(x, dim, state);
        case 8:  return gpu_F8(x, dim);
        case 9:  return gpu_F9(x, dim);
        case 10: return gpu_F10(x, dim);
        case 11: return gpu_F11(x, dim);
        default: return INFINITY;
    }
}

/* ---------------- CUDA Kernels ---------------- */

__global__ void setup_rand_kernel(curandState* state, unsigned long long seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void init_population_kernel(
    double* X, double* fitness,
    const double* lb, const double* ub,
    int Coatis, int DIMENSION, int F, curandState* states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Coatis) return;

    curandState local_state = states[i];

    for (int j = 0; j < DIMENSION; j++) {
        double r = curand_uniform_double(&local_state);
        X[i * DIMENSION + j] = lb[j] + r * (ub[j] - lb[j]);
    }

    fitness[i] = gpu_evaluation_fitness(X + i * DIMENSION, F, DIMENSION, &local_state);

    states[i] = local_state;
}

__global__ void find_best_coati_kernel(const double* fitness, int n, double* out_val, int* out_idx) {
    extern __shared__ Element s_data[];

    int tid = threadIdx.x;
    if (tid < n) {
        s_data[tid].val = fitness[tid];
        s_data[tid].idx = tid;
    } else {
        s_data[tid].val = INFINITY;
        s_data[tid].idx = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s].val < s_data[tid].val) {
                s_data[tid] = s_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_val = s_data[0].val;
        *out_idx = s_data[0].idx;
    }
}

__global__ void exploitative_phase_kernel(
    double* X, double* fitness, const double* Iguana,
    const double* lb, const double* ub,
    int Coatis, int DIMENSION, int F, curandState* states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Coatis) return;

    curandState local_state = states[i];
    double coati_new_position[30];
    
    if (i < Coatis / 2) {
        // First half coatis attack Iguana
        for (int j = 0; j < DIMENSION; j++) {
            double r = curand_uniform_double(&local_state);
            double coin = curand_uniform_double(&local_state);
            int I = (coin < 0.5) ? 1 : 2;
            
            coati_new_position[j] = X[i * DIMENSION + j] + r * (Iguana[j] - (I * X[i * DIMENSION + j]));
            coati_new_position[j] = fmax(lb[j], fmin(coati_new_position[j], ub[j]));
        }

        double fitness_new = gpu_evaluation_fitness(coati_new_position, F, DIMENSION, &local_state);
        double fitness_old = fitness[i];
        if (fitness_new < fitness_old) {
            for (int j = 0; j < DIMENSION; j++) {
                X[i * DIMENSION + j] = coati_new_position[j];
            }
            fitness[i] = fitness_new;
        }
    }
    else {
        // Second half coatis attack fallen Iguana (IguanaG)
        double IguanaG[30];
        for (int j = 0; j < DIMENSION; j++) {
            double r = curand_uniform_double(&local_state);
            IguanaG[j] = lb[j] + r * (ub[j] - lb[j]);
        }

        double fitness_IguanaG = gpu_evaluation_fitness(IguanaG, F, DIMENSION, &local_state);
        double fitness_Coati = fitness[i];

        if (fitness_IguanaG < fitness_Coati) {
            for (int j = 0; j < DIMENSION; j++) {
                double r = curand_uniform_double(&local_state);
                double coin = curand_uniform_double(&local_state);
                int I = (coin < 0.5) ? 1 : 2;
                
                coati_new_position[j] = X[i * DIMENSION + j] + r * (IguanaG[j] - (I * X[i * DIMENSION + j]));
                coati_new_position[j] = fmax(lb[j], fmin(coati_new_position[j], ub[j]));
            }
        }
        else {
            for (int j = 0; j < DIMENSION; j++) {
                double r = curand_uniform_double(&local_state);
                coati_new_position[j] = X[i * DIMENSION + j] + r * (X[i * DIMENSION + j] - IguanaG[j]);
                coati_new_position[j] = fmax(lb[j], fmin(coati_new_position[j], ub[j]));
            }
        }

        double fitness_new = gpu_evaluation_fitness(coati_new_position, F, DIMENSION, &local_state);
        if (fitness_new < fitness_Coati) {
            for (int j = 0; j < DIMENSION; j++) {
                X[i * DIMENSION + j] = coati_new_position[j];
            }
            fitness[i] = fitness_new;
        }
    }

    states[i] = local_state;
}

__global__ void compute_local_bounds_kernel(
    double* lbLocal, double* ubLocal,
    const double* lb, const double* ub,
    int DIMENSION, int iteration)
{
    int j = threadIdx.x;
    if (j >= DIMENSION) return;

    double t = iteration + 1;
    double lbl = lb[j] / t;
    double ubl = ub[j] / t;

    lbl = fmax(lb[j], fmin(lbl, ub[j]));
    ubl = fmax(lb[j], fmin(ubl, ub[j]));

    if (lbl > ubl) {
        double temp = lbl;
        lbl = ubl;
        ubl = temp;
    }

    lbLocal[j] = lbl;
    ubLocal[j] = ubl;
}

__global__ void exploration_phase_kernel(
    double* X, double* fitness,
    const double* lbLocal, const double* ubLocal,
    const double* lb, const double* ub,
    int Coatis, int DIMENSION, int F, curandState* states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Coatis) return;

    curandState local_state = states[i];
    double coati_new_position[30];

    for (int j = 0; j < DIMENSION; j++) {
        double r = curand_uniform_double(&local_state);
        coati_new_position[j] = X[i * DIMENSION + j] + (1 - 2 * r) * (lbLocal[j] + r * (ubLocal[j] - lbLocal[j]));
        coati_new_position[j] = fmax(lb[j], fmin(coati_new_position[j], ub[j]));
    }

    double fitness_new = gpu_evaluation_fitness(coati_new_position, F, DIMENSION, &local_state);
    double fitness_old = fitness[i];
    if (fitness_new < fitness_old) {
        for (int j = 0; j < DIMENSION; j++) {
            X[i * DIMENSION + j] = coati_new_position[j];
        }
        fitness[i] = fitness_new;
    }

    states[i] = local_state;
}

/* ---------------- Host Functions ---------------- */

int get_next_power_of_2(int n) {
    int val = 1;
    while (val < n) {
        val <<= 1;
    }
    return val;
}

int main(int argc, char** argv) {
    int F{1}, Coatis{30}, Max_iterations{500}, Run{1};
    if (argc >= 2) F = atoi(argv[1]);
    if (argc >= 3) Coatis = atoi(argv[2]);
    if (argc >= 4) Max_iterations = atoi(argv[3]);
    if (argc >= 5) Run = atoi(argv[4]);

    BenchmarkInfo info = get_benchmark_info(F);
    int DIMENSION = info.dim;
    vector<double> lb = info.lb;
    vector<double> ub = info.ub;

    vector<double> best_score_log;
    vector<int> iter_log;
    best_score_log.reserve(Max_iterations / 10 + 1);
    iter_log.reserve(Max_iterations / 10 + 1);

    auto start_time = chrono::high_resolution_clock::now();

    // Device allocation
    double* d_X;
    double* d_fitness;
    double* d_lb;
    double* d_ub;
    double* d_Iguana;
    double* d_lbLocal;
    double* d_ubLocal;
    curandState* d_states;

    double* d_best_val;
    int* d_best_idx;

    CUDA_CHECK(cudaMalloc(&d_X, Coatis * DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitness, Coatis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lb, DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ub, DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Iguana, DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lbLocal, DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ubLocal, DIMENSION * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_states, Coatis * sizeof(curandState)));

    CUDA_CHECK(cudaMalloc(&d_best_val, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));

    // Copy bounds to device
    CUDA_CHECK(cudaMemcpy(d_lb, lb.data(), DIMENSION * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ub, ub.data(), DIMENSION * sizeof(double), cudaMemcpyHostToDevice));

    // Launch random generator setup
    int threads_per_block = 256;
    int blocks = (Coatis + threads_per_block - 1) / threads_per_block;
    setup_rand_kernel<<<blocks, threads_per_block>>>(d_states, 1234ULL, Coatis);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize population
    init_population_kernel<<<blocks, threads_per_block>>>(d_X, d_fitness, d_lb, d_ub, Coatis, DIMENSION, F, d_states);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find initial best
    int reduction_threads = get_next_power_of_2(Coatis);
    find_best_coati_kernel<<<1, reduction_threads, reduction_threads * sizeof(Element)>>>(d_fitness, Coatis, d_best_val, d_best_idx);
    CUDA_CHECK(cudaDeviceSynchronize());

    double host_best_val;
    int host_best_idx;
    CUDA_CHECK(cudaMemcpy(&host_best_val, d_best_val, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));

    for (int iteration = 0; iteration < Max_iterations; iteration++) {
        // Set Iguana position (device-to-device copy of the best coati)
        CUDA_CHECK(cudaMemcpy(d_Iguana, d_X + host_best_idx * DIMENSION, DIMENSION * sizeof(double), cudaMemcpyDeviceToDevice));

        // Exploitative Phase
        exploitative_phase_kernel<<<blocks, threads_per_block>>>(d_X, d_fitness, d_Iguana, d_lb, d_ub, Coatis, DIMENSION, F, d_states);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Local bounds computation
        compute_local_bounds_kernel<<<1, DIMENSION>>>(d_lbLocal, d_ubLocal, d_lb, d_ub, DIMENSION, iteration);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exploration Phase
        exploration_phase_kernel<<<blocks, threads_per_block>>>(d_X, d_fitness, d_lbLocal, d_ubLocal, d_lb, d_ub, Coatis, DIMENSION, F, d_states);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Find iteration best
        find_best_coati_kernel<<<1, reduction_threads, reduction_threads * sizeof(Element)>>>(d_fitness, Coatis, d_best_val, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&host_best_val, d_best_val, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&host_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));

        if (iteration % std::max(1, Max_iterations / 10) == 0) {
            iter_log.push_back(iteration);
            best_score_log.push_back(host_best_val);
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto execution_time = chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Log to file
    std::string log_filename =
        "logs/cuda/F" + std::to_string(F) +
        "_C" + std::to_string(Coatis) +
        "_I" + std::to_string(Max_iterations) +
        "_R" + std::to_string(Run) + ".log";

    std::ofstream log(log_filename, std::ios::out);
    if (log) {
        log << "RUN "
            << "F=" << F
            << " COATIS=" << Coatis
            << " ITER=" << Max_iterations
            << " IMPL=CUDA\n";

        for (size_t i = 0; i < iter_log.size(); ++i) {
            log << "ITER "
                << iter_log[i] << " "
                << best_score_log[i] << "\n";
        }

        log << "TIME_MS " << execution_time << "\n";
        log << "BEST_SCORE " << host_best_val << "\n";
        log << "END_RUN\n";
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_fitness);
    cudaFree(d_lb);
    cudaFree(d_ub);
    cudaFree(d_Iguana);
    cudaFree(d_lbLocal);
    cudaFree(d_ubLocal);
    cudaFree(d_states);
    cudaFree(d_best_val);
    cudaFree(d_best_idx);

    return 0;
}