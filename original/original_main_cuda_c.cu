#include <bits/stdc++.h>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

using Vec = std::vector<double>;

inline double PI() { return 3.14159265358979323846; }

/* ---------------- Ufun helper ---------------- */
inline double Ufun(const Vec &x, double a, double k, double m) {
    double R = 0.0;
    for (double xi : x) {
        if (xi > a) {
            R += k * std::pow(xi - a, m);
        } else if (xi < -a) {
            R += k * std::pow(-xi - a, m);
        }
    }
    return R;
}

/* ---------------- F1–F23 (CPU) ---------------- */

// F1
inline double F1(const Vec &x) {
    double R = 0.0;
    for (double v : x) R += v * v;
    return R;
}

// F2
inline double F2(const Vec &x) {
    double s = 0.0, p = 1.0;
    for (double v : x) {
        double a = std::fabs(v);
        s += a;
        p *= a;
    }
    return s + p;
}

// F3
inline double F3(const Vec &x) {
    int dim = (int)x.size();
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        double sum_i = 0.0;
        for (int j = 0; j <= i; ++j) sum_i += x[j];
        R += sum_i * sum_i;
    }
    return R;
}

// F4
inline double F4(const Vec &x) {
    double mx = 0.0;
    for (double v : x) mx = std::max(mx, std::fabs(v));
    return mx;
}

// F5 (Rosenbrock)
inline double F5(const Vec &x) {
    int dim = (int)x.size();
    double R = 0.0;
    for (int i = 0; i < dim - 1; ++i) {
        double t1 = x[i+1] - x[i]*x[i];
        double t2 = x[i] - 1.0;
        R += 100.0 * t1 * t1 + t2 * t2;
    }
    return R;
}

// F6
inline double F6(const Vec &x) {
    double R = 0.0;
    for (double v : x) {
        double t = std::floor(v + 0.5);
        R += t * t;
    }
    return R;
}

// F7 
inline double F7(const Vec &x) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int dim = (int)x.size();
    double R = 0.0;
    for (int i = 0; i < dim; ++i) {
        R += (i + 1) * std::pow(x[i], 4);
    }
    R += dist(rng); // + rand
    return R;
}

// F8
inline double F8(const Vec &x) {
    double R = 0.0;
    for (double v : x) {
        R += -v * std::sin(std::sqrt(std::fabs(v)));
    }
    return R;
}

// F9 (Rastrigin)
inline double F9(const Vec &x) {
    int dim = (int)x.size();
    double R = 0.0;
    for (double v : x) {
        R += v*v - 10.0 * std::cos(2.0 * PI() * v);
    }
    return R + 10.0 * dim;
}

// F10 (Ackley)
inline double F10(const Vec &x) {
    int dim = (int)x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (double v : x) {
        sum1 += v*v;
        sum2 += std::cos(2.0 * PI() * v);
    }
    return -20.0 * std::exp(-0.2 * std::sqrt(sum1 / dim))
           - std::exp(sum2 / dim) + 20.0 + std::exp(1.0);
}

// F11 (Griewank)
inline double F11(const Vec &x) {
    int dim = (int)x.size();
    double sum2 = 0.0, prod = 1.0;
    for (int i = 0; i < dim; ++i) {
        double v = x[i];
        sum2 += v*v / 4000.0;
        prod *= std::cos(v / std::sqrt(double(i+1)));
    }
    return sum2 - prod + 1.0;
}

/* --------- wrapper: choose fitness by F index (1..11) --------- */

inline double evaluation_fitness(const Vec &x, int F) {
    switch (F) {
        case 1:  return F1(x);
        case 2:  return F2(x);
        case 3:  return F3(x);
        case 4:  return F4(x);
        case 5:  return F5(x);
        case 6:  return F6(x);
        case 7:  return F7(x);
        case 8:  return F8(x);
        case 9:  return F9(x);
        case 10: return F10(x);
        case 11: return F11(x);
        default: throw std::runtime_error("Unknown F index");
    }
}

/* --------- bounds/dimension (from fun_info.m) --------- */

struct BenchmarkInfo {
    std::vector<double> lb;
    std::vector<double> ub;
    int dim;
};

inline BenchmarkInfo get_benchmark_info(int F) {
    BenchmarkInfo info{};
    switch (F) {
    case 1:  info.dim=30; info.lb.assign(info.dim,-100); info.ub.assign(info.dim,100); break;
    case 2:  info.dim=30; info.lb.assign(info.dim,-10);  info.ub.assign(info.dim,10);  break;
    case 3:  info.dim=30; info.lb.assign(info.dim,-100); info.ub.assign(info.dim,100); break;
    case 4:  info.dim=30; info.lb.assign(info.dim,-100); info.ub.assign(info.dim,100); break;
    case 5:  info.dim=30; info.lb.assign(info.dim,-30);  info.ub.assign(info.dim,30);  break;
    case 6:  info.dim=30; info.lb.assign(info.dim,-100); info.ub.assign(info.dim,100); break;
    case 7:  info.dim=30; info.lb.assign(info.dim,-1.28);info.ub.assign(info.dim,1.28);break;
    case 8:  info.dim=30; info.lb.assign(info.dim,-500); info.ub.assign(info.dim,500); break;
    case 9:  info.dim=30; info.lb.assign(info.dim,-5.12);info.ub.assign(info.dim,5.12);break;
    case 10: info.dim=30; info.lb.assign(info.dim,-32);  info.ub.assign(info.dim,32);  break;
    case 11: info.dim=30; info.lb.assign(info.dim,-600); info.ub.assign(info.dim,600); break;
    default: throw std::runtime_error("Unknown F index in get_benchmark_info");
    }
    return info;
}

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
            lbLocal[j] = max(lb[j], lb[j]/t);
            ubLocal[j] = min(ub[j], ub[j]/t);

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