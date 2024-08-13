#include <chrono>
#include <iostream>
#include <vector>
#include <thread>

const int NUM_THREADS[7] = {2, 4, 7, 8, 16, 20, 40};
int thread_idx;

int M = 20000;
int N = 20000;
const int T = 10;

void init(double* a, double* b, double* c, int m, int n);
void matrix_vector_product(const double* a, const double* b, double* c, int m, int n);

void init_parallel(double* a, double* b, double* c, int n, int lb, int ub);
void matrix_vector_product_parallel(const double* a, const double* b, double* c, int n, int lb, int ub);

int main() {
    for (int m = 1; m <= 2; ++m) {
        M *= m;
        N *= m;
        std::cout << "THE RESULTS FOR " << M << " INPUT DATA:" << std::endl;
        auto a = std::make_unique<double[]>(M * N);
        auto b = std::make_unique<double[]>(N);
        auto c = std::make_unique<double[]>(M);

        std::cout.precision(2);

        std::chrono::duration<double> total_serial = std::chrono::duration<double>::zero();
        for (int i = 0; i < T; ++i) {
            auto start{std::chrono::steady_clock::now()};
            init(a.get(), b.get(), c.get(), M, N);
            matrix_vector_product(a.get(), b.get(), c.get(), M, N);
            auto end{std::chrono::steady_clock::now()};
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_serial += elapsed_seconds;
        }
        std::cout << "\tThe serial calculations takes about " << total_serial.count() / T << " seconds!" << std::endl;

        for (thread_idx = 0; thread_idx < 7; ++thread_idx) {
            std::cout << "\tTHE RESULTS FOR " << NUM_THREADS[thread_idx] << " THREADS:" << std::endl;
            std::chrono::duration<double> total_parallel = std::chrono::duration<double>::zero();
            for (int i = 0; i < T; ++i) {
                std::vector<std::jthread> threads;
                auto start = std::chrono::steady_clock::now();
                int items_per_thread = M / NUM_THREADS[thread_idx];
                for (int th = 1; th <= NUM_THREADS[thread_idx]; ++th)
                    threads.emplace_back(init_parallel, a.get(), b.get(), c.get(), N,
                                         (th - 1) * items_per_thread, th * items_per_thread - 1);
                threads.clear();
                for (int th = 1; th <= NUM_THREADS[thread_idx]; ++th)
                    threads.emplace_back(matrix_vector_product_parallel, a.get(), b.get(), c.get(), N,
                                         (th - 1) * items_per_thread, th * items_per_thread - 1);
                threads.clear();
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                total_parallel += elapsed_seconds;
            }
            std::cout << "\tThe parallel calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
            std::cout << "\t" << total_serial / total_parallel << "x acceleration" << std::endl;
            std::cout << "\t-----------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    }
    return EXIT_SUCCESS;
}

void init(double* a, double* b, double* c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            a[i * n + j] = i + j;
        c[i] = 0.0;
    }
    for (int j = 0; j < n; ++j)
        b[j] = j;
}

void matrix_vector_product(const double* a, const double* b, double* c, const int m, const int n) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
}

void init_parallel(double* a, double* b, double* c, int n, int lb, int ub) {
    for (int i = lb; i <= ub; ++i) {
        for (int j = 0; j < n; ++j)
            a[i * n + j] = i + j;
        c[i] = 0.0;
    }
    for (int j = lb; j <= ub; ++j)
        b[j] = j;
}

void matrix_vector_product_parallel(const double* a, const double* b, double* c, const int n, int lb, int ub) {
    for (int i = lb; i <= ub; ++i)
        for (int j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
}
