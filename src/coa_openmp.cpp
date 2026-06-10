#include <bits/stdc++.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "../include/benchmark_functions.h"

using namespace std;

struct COAResult
{
    double best_score;
    int best_coati_index;
    vector<double> best_coati;
    COAResult(double score, int initialSize) : best_score(score), best_coati(initialSize), best_coati_index(-1) {}
};

// Thread-safe random number generation
static thread_local std::mt19937_64 rng_engine;
double urand(){
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_engine);
}
int urandI(){
    static thread_local std::uniform_int_distribution<int> coin(1, 2);
    return coin(rng_engine);
}


COAResult COA_openmp(int F, int Coatis, int Max_iterations, vector<double>& best_score_log, vector<int>& iter_log)
{
    BenchmarkInfo info = get_benchmark_info(F);
    int DIMENSION = info.dim;
    vector<double> lb = info.lb;
    vector<double> ub = info.ub;
    COAResult result(INFINITY, DIMENSION);

    vector<vector<double>> X(Coatis, vector<double>(DIMENSION));
    vector<double> current_fitness(Coatis, INFINITY);

    vector<double> Iguana(DIMENSION);
    vector<double> lbLocal(DIMENSION, 0);
    vector<double> ubLocal(DIMENSION, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 local_rng;
        {
            std::mt19937_64 seed_gen(std::random_device{}());
            local_rng.seed(seed_gen() + 0x9e3779b97f4a7c15ULL * tid);
        }
        std::uniform_real_distribution<double> local_dist(0.0, 1.0);
        std::uniform_int_distribution<int> local_coin(1, 2);
    
        double local_best_score = std::numeric_limits<double>::infinity();
        int local_best_index = -1;

        vector<double> coati_new_position(DIMENSION, 0.0);
        vector<double> IguanaG(DIMENSION, 0.0);

        // Initialize population
        #pragma omp for
        for (int i = 0; i < Coatis; i++)
        {
            for (int j = 0; j < DIMENSION; ++j)
            {   
                double r = local_dist(local_rng);
                X[i][j] = lb[j] + r * (ub[j] - lb[j]);
            }
        }

        // Evaluate initial fitness and find the best coati
        #pragma omp for 
        for (int i = 0; i < Coatis; ++i)
        {
            double fitness = evaluation_fitness(X[i], F);
            current_fitness[i] = fitness;
            if (fitness < local_best_score)
            {
                local_best_score = fitness;
                local_best_index = i;
            }
        }

        // Merge local best into global best (once per thread)
        #pragma omp critical
        {
            if (local_best_score < result.best_score)
            {
                result.best_score = local_best_score;
                result.best_coati_index = local_best_index;
            }
        }

        #pragma omp barrier

        for (int iteration = 0; iteration < Max_iterations; iteration++)
        {
            #pragma omp single
            {
                Iguana = X[result.best_coati_index];
            } // Implicit barrier: all threads wait until Iguana is updated

            //--------------- Exploitative phase (Global search) ----------------//
            
            // Using First half coati population to attack the Iguana
            #pragma omp for nowait
            for (int i = 0; i < Coatis / 2; i++)
            {
                for (int j = 0; j < DIMENSION; j++)
                {
                    int I = local_coin(local_rng);
                    double r = local_dist(local_rng);
                    
                    coati_new_position[j] = X[i][j] + r * (Iguana[j] - (I * X[i][j]));
                    coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
                }

                // Updating coati and best solution if the new position is better
                double fitness_new = evaluation_fitness(coati_new_position, F);
                double fitness_old = current_fitness[i];
                if (fitness_new < fitness_old)
                {
                    X[i] = coati_new_position;
                    current_fitness[i] = fitness_new;
                }
            }
        
            // Using Second half coati population to attack the fallen Iguana - IguanaG (explore new solutions)
            #pragma omp for
            for (int i = Coatis / 2; i < Coatis; i++)
            {
                for (int j = 0; j < DIMENSION; j++)
                {
                    double r = local_dist(local_rng);
                    IguanaG[j] = lb[j] + r * (ub[j] - lb[j]);
                }
    
                double fitness_IguanaG = evaluation_fitness(IguanaG, F);
                double fitness_Coati = current_fitness[i];
    
                if (fitness_IguanaG < fitness_Coati)
                {
                    for (int j = 0; j < DIMENSION; j++)
                    {
                        double r = local_dist(local_rng);
                        int I = local_coin(local_rng);
                        coati_new_position[j] = X[i][j] + r * (IguanaG[j] - (I * X[i][j]));
                        coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
                    }
                }
                else
                {
                    for (int j = 0; j < DIMENSION; j++)
                    {
                        double r = local_dist(local_rng);
                        coati_new_position[j] = X[i][j] + r * (X[i][j] - IguanaG[j]);
                        coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
                    }
                }
    
                double fitness_new = evaluation_fitness(coati_new_position, F);
                if (fitness_new < fitness_Coati)
                {
                    X[i] = coati_new_position;
                    current_fitness[i] = fitness_new;
                }
            } // Implicit barrier: all threads wait until exploitative phase completes
        
            //--------------- Exploration phase (Local search) ----------------//
            double t = iteration + 1;

            #pragma omp for
            for (int j = 0; j < DIMENSION; j++)
            {
                lbLocal[j] = lb[j] / t;
                ubLocal[j] = ub[j] / t;

                lbLocal[j] = std::clamp(lbLocal[j], lb[j], ub[j]);
                ubLocal[j] = std::clamp(ubLocal[j], lb[j], ub[j]);

                if (lbLocal[j] > ubLocal[j])
                {
                    swap(lbLocal[j], ubLocal[j]);
                }
            } // Implicit barrier: all threads wait until bounds are updated
        
            // Updating all coatis based on local bounds
            #pragma omp for
            for (int i = 0; i < Coatis; i++)
            {
                for (int j = 0; j < DIMENSION; j++)
                {
                    double r = local_dist(local_rng);
                    coati_new_position[j] = X[i][j] + (1 - 2 * r) * (lbLocal[j] + r * (ubLocal[j] - lbLocal[j]));
                    coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
                }

                double fitness_new = evaluation_fitness(coati_new_position, F);
                double fitness_old = current_fitness[i];
                if (fitness_new < fitness_old)
                {
                    X[i] = coati_new_position;
                    current_fitness[i] = fitness_new;
                }
            } // Implicit barrier: all threads wait until local search completes
            
            #pragma omp single
            {
                double best_iter_score = std::numeric_limits<double>::infinity();
                int best_iter_index = -1;

                for (int i = 0; i < Coatis; i++){
                    if (current_fitness[i] < best_iter_score)
                    {
                        best_iter_score = current_fitness[i];
                        best_iter_index = i;
                    }
                }

                result.best_score = best_iter_score;
                result.best_coati_index = best_iter_index;

                if (iteration % std::max(1, Max_iterations / 10) == 0) {
                    iter_log.push_back(iteration);
                    best_score_log.push_back(result.best_score);
                }
            } // Implicit barrier: all threads wait until logs are updated
        }
    }
    
    result.best_coati = X[result.best_coati_index];
    return result;
}

int main(int argc, char **argv)
{
    int F{1}, Coatis{30}, Max_iterations{500}, Run{1};
    if (argc >= 2)
        F = atoi(argv[1]);
    if (argc >= 3)
        Coatis = atoi(argv[2]);
    if (argc >= 4)
        Max_iterations = atoi(argv[3]);
    if (argc >= 5)
        Run = atoi(argv[4]);

    // Logs
    vector<double> best_score_log;
    vector<int> iter_log;
    
    best_score_log.reserve(Max_iterations / 10 + 1);
    iter_log.reserve(Max_iterations / 10 + 1);
        
    auto start = std::chrono::high_resolution_clock::now();
    auto result = COA_openmp(F, Coatis, Max_iterations, best_score_log, iter_log);
    auto end = std::chrono::high_resolution_clock::now();
    auto execution_time = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // ----------------- File Logging -----------------
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Construct per-run log filename
    std::string log_filename =
        "logs/openmp/F" + std::to_string(F) +
        "_C" + std::to_string(Coatis) +
        "_I" + std::to_string(Max_iterations) +
        "_T" + std::to_string(num_threads) +
        "_R" + std::to_string(Run) + ".log";

    // Open file (overwrite per run — correct)
    std::ofstream log(log_filename, std::ios::out);
    if (!log) {
        std::cerr << "ERROR: Cannot open log file: " << log_filename << "\n";
        return 1;
    }

    // --------- Write machine-parsable log ---------
    log << "RUN "
        << "F=" << F
        << " COATIS=" << Coatis
        << " ITER=" << Max_iterations
        << " IMPL=OpenMP\n";

    for (size_t i = 0; i < iter_log.size(); ++i) {
        log << "ITER "
            << iter_log[i] << " "
            << best_score_log[i] << "\n";
    }

    log << "TIME_MS " << execution_time << "\n";
    log << "BEST_SCORE " << result.best_score << "\n";
    log << "END_RUN\n";


    return 0;
}