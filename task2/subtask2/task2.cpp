#include <chrono>
#include <iostream>
#include <cmath>
#include <omp.h>

const int NUM_THREADS[7] = {2, 4, 7, 8, 16, 20, 40};
int thread_idx;

const double A = -4;
const double B = 4;
const int N_STEPS = 40000000;
const int T = 10;

double func(double x) {
    return exp(-x * x);
}

double integrate_serial(double (*func)(double), double a, double b, int n);
double integrate_omp(double (*func)(double), double a, double b, int n);

int main() {
    for (thread_idx = 0; thread_idx < 7; ++thread_idx) {
        std::cout << "THE RESULTS FOR " << NUM_THREADS[thread_idx] << " THREADS:" << std::endl;

        std::chrono::duration<double> total_serial = std::chrono::duration<double>::zero();
        double total_error = 0;
        for (int i = 0; i < T; ++i) {
            auto start{std::chrono::steady_clock::now()};
            double res = integrate_serial(func, A, B, N_STEPS);
            auto end{std::chrono::steady_clock::now()};
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_serial += elapsed_seconds;
            total_error += fabs(M_PI - res);
        }
        std::cout.precision(2);
        std::cout << "\tThe serial calculations takes about " << total_serial.count() / T << " seconds!" << std::endl;
        std::cout.precision(10);
        std::cout << "\tError = ";
        std::cout << total_error / T << std::endl;

        std::chrono::duration<double> total_parallel = std::chrono::duration<double>::zero();
        total_error = 0;
        for (int i = 0; i < T; ++i) {
            auto start = std::chrono::steady_clock::now();
            double res = integrate_omp(func, A, B, N_STEPS);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_parallel += elapsed_seconds;
            total_error += fabs(M_PI - res);
        }
        std::cout.precision(2);
        std::cout << "\tThe parallel calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
        std::cout.precision(10);
        std::cout << "\tError = " << total_error / T << std::endl;

        std::cout.precision(2);
        std::cout << "\t" << total_serial / total_parallel << "x acceleration" << std::endl;
        std::cout << "-----------------------------------------------------------------------------" << std::endl;
    }
    return EXIT_SUCCESS;
}

double integrate_serial(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0;
    for (int i = 0; i < n; ++i)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0;
    #pragma omp parallel num_threads(NUM_THREADS[thread_idx])
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sum_loc = 0;

        for (int i = lb; i <= ub; ++i)
            sum_loc += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sum_loc;
    }
    sum *= h;
    return sum;
}
