#include <bits/stdc++.h>
#include <random>
#include <chrono>
#include "./_benchmark_functions.h"

using namespace std;

struct COAResult {
    double best_score;
    vector<double> best_positions;
    COAResult(double score, int initialSize) : best_score(score), best_positions(initialSize) {} 
};

static std::mt19937_64 rng{std::random_device{}()};
double urand(double a, double b) {
    std::uniform_real_distribution<double> dist(a,b);
    return dist(rng);
}
int urandI(int a, int b) {    
    std::uniform_int_distribution<int> coin(a,b);
    return coin(rng);
}



COAResult COA_sequential(int F, int Coatis, int Max_iterations) {
    BenchmarkInfo info = get_benchmark_info(F);
    int DIMENSION = info.dim;
    vector<double> lb = info.lb;
    vector<double> ub = info.ub;
    COAResult result(INFINITY, DIMENSION);

    cout << "Dimension: " << DIMENSION << ", Bounds: [" << lb[0] << "," << ub[0] << "]\n";

    vector<vector<double>> X(Coatis, vector<double>(DIMENSION));

    // Initialize population
    for(int i = 0; i < Coatis; i++) {
        for (int j = 0; j < DIMENSION; ++j) {
            double r = urand(0.0, 1.0);
            X[i][j] = lb[j] + r * (ub[j] - lb[j]);
        }
    }
    
    // Evaluate initial fitness and find the best coati
    for(int i = 0; i < Coatis; ++i) {
        double fitness = evaluation_fitness(X[i], F);
        if(fitness < result.best_score){
            result.best_score = fitness;
            result.best_positions = X[i];
        }
    }
    
    //Setting Iguana as the best Coati
    vector<double> Iguana(DIMENSION);
    
    
    for(int iteration=0; iteration<Max_iterations; iteration++){
        Iguana = result.best_positions;

        //--------------- Exploitative phase (Global search) ----------------//

        // Using First half coati population to attack the Iguana
        for(int i=0; i<Coatis/2; i++){
            vector<double> coati_new_position(DIMENSION);
            for(int j=0; j<DIMENSION; j++){
                double r = urand(0.0, 1.0);
                int I = urandI(1,2);
                coati_new_position[j] = X[i][j] +  r * (Iguana[j] - (I * X[i][j]));
                coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
            }

            // Updating coati and best solution if the new position is better
            double fitness_new = evaluation_fitness(coati_new_position, F);
            double fitness_old = evaluation_fitness(X[i], F);
            if(fitness_new < fitness_old) {
                X[i] = coati_new_position;
            }
            if(fitness_new < result.best_score){
                result.best_score = fitness_new;
                result.best_positions = coati_new_position;
            }
        
        }
        // Using Second half coati population to attack the fallen Iguana - IguanaG (explore new solutions)
        for(int i=Coatis/2; i<Coatis; i++){
            vector<double> coati_new_position(DIMENSION);

            // Calculating IguanaG randomly within bounds
            vector<double> IguanaG(DIMENSION);
            for(int j=0; j<DIMENSION; j++){
                double r = urand(0.0, 1.0);
                IguanaG[j] = lb[j] + r * (ub[j] - lb[j]);
            }

            double fitness_IguanaG = evaluation_fitness(IguanaG, F);
            double fitness_Coati = evaluation_fitness(X[i], F);

            if(fitness_IguanaG < fitness_Coati){
                for(int j=0; j<DIMENSION; j++){
                    double r = urand(0.0, 1.0);
                    int I = urandI(1,2);
                    coati_new_position[j] = X[i][j] + r * (IguanaG[j] - (I * X[i][j]));
                    coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
                }
            }
            else{
                for(int j=0; j<DIMENSION; j++){
                    double r = urand(0.0, 1.0);
                    coati_new_position[j] = X[i][j] + r*(X[i][j] - IguanaG[j]);
                    coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);    
                }
            }

            double fitness_new = evaluation_fitness(coati_new_position, F);
            if(fitness_new < fitness_Coati) {
                X[i] = coati_new_position;
            }
            if(fitness_new < result.best_score){
                result.best_score = fitness_new;
                result.best_positions = coati_new_position;
            }

        }
        
        //--------------- Exploration phase (Local search) ----------------//

        // Defining local bounds for each dimension
        vector<double> lbLocal(DIMENSION);
        vector<double> ubLocal(DIMENSION);
        double t = iteration + 1;
        for(int j=0; j<DIMENSION; j++){
            lbLocal[j] = lb[j]/t;
            ubLocal[j] = ub[j]/t;

            lbLocal[j] = std::clamp(lbLocal[j], lb[j], ub[j]);
            ubLocal[j] = std::clamp(ubLocal[j], lb[j], ub[j]);
            
            if(lbLocal[j] > ubLocal[j]){
                swap(lbLocal[j], ubLocal[j]);
            }
        }

        // Updating all coatis based on local bounds
        for(int i=0; i<Coatis; i++){
            vector<double> coati_new_position(DIMENSION);
            for(int j=0; j<DIMENSION; j++){
                double r = urand(0.0, 1.0);
                coati_new_position[j] = X[i][j] + (1 - 2 * r) * (lbLocal[j] + r * (ubLocal[j] - lbLocal[j]));
                coati_new_position[j] = std::clamp(coati_new_position[j], lb[j], ub[j]);
            }

            double fitness_new = evaluation_fitness(coati_new_position, F);
            double fitness_old = evaluation_fitness(X[i], F);
            if(fitness_new < fitness_old) {
                X[i] = coati_new_position;
            }
            if(fitness_new < result.best_score){
                result.best_score = fitness_new;
                result.best_positions = coati_new_position;
            }
        }

        if (iteration % std::max(1, Max_iterations/10) == 0) {
            std::cout << "[iter " << iteration << "] best = " << result.best_score << "\n";
        }
    }
    return result;
}

int main(int argc, char** argv) {
    int F{1}, Coatis{30},Max_iterations{500};
    if (argc >= 2) F = atoi(argv[1]);
    if (argc >= 3) Coatis= atoi(argv[2]);
    if (argc >= 4) Max_iterations = atoi(argv[3]);

    cout << "COA Sequential, F" << F << ", Coatis=" << Coatis << ", Max_iterations=" << Max_iterations << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    auto result = COA_sequential(F, Coatis, Max_iterations);
    auto end = std::chrono::high_resolution_clock::now();
    auto execution_time = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if(execution_time < 1000)
        cout << "Elapsed time: " << execution_time << " milliseconds\n";
    else
        cout << "Elapsed time: " << execution_time / 1000.0 << " seconds\n";

    cout << "Best score = " << result.best_score << "\n";
    cout << "Best position (first few dims): ";
    for (int i = 0; i < result.best_positions.size(); i++)
        cout << result.best_positions[i] << " ";
    cout << endl;

    return 0;
}