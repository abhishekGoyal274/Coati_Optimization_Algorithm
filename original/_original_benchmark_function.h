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