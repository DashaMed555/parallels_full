#include <chrono>
#include <iostream>
#include <omp.h>

const int NUM_THREADS[7] = {2, 4, 7, 8, 16, 20, 40};
int thread_idx;

int M = 20000;
int N = 20000;
const int T = 10;

void init_serial(double *a, double *b, double *c, int m, int n);
void init_omp(double *a, double *b, double *c, int m, int n);
void matrix_vector_product_serial(const double *a, const double *b, double *c, int m, int n);
void matrix_vector_product_omp(const double *a, const double *b, double *c, int m, int n);

int main() {
    for (int m = 1; m <= 2; ++m) {
        M *= m;
        N *= m;
        std::cout << "THE RESULTS FOR " << M << " INPUT DATA:" << std::endl;
        auto a = (double*)malloc(sizeof(double) * M * N);
        if (a == nullptr) {
            perror("");
            return EXIT_FAILURE;
        }
        auto b = (double*)malloc(sizeof(double) * N);
        if (b == nullptr) {
            perror("");
            free(a);
            return EXIT_FAILURE;
        }
        auto c = (double*)malloc(sizeof(double) * M);
        if (c == nullptr) {
            perror("");
            free(a);
            free(b);
            return EXIT_FAILURE;
        }

        std::cout.precision(2);

        for (thread_idx = 0; thread_idx < 7; ++thread_idx) {
            std::cout << "\tTHE RESULTS FOR " << NUM_THREADS[thread_idx] << " THREADS:" << std::endl;

            std::chrono::duration<double> total_serial = std::chrono::duration<double>::zero();
            for (int i = 0; i < T; ++i) {
                auto start{std::chrono::steady_clock::now()};
                init_serial(a, b, c, M, N);
                matrix_vector_product_serial(a, b, c, M, N);
                auto end{std::chrono::steady_clock::now()};
                std::chrono::duration<double> elapsed_seconds = end - start;
                total_serial += elapsed_seconds;
            }
            std::cout << "\tThe serial calculations takes about " << total_serial.count() / T << " seconds!" << std::endl;

            std::chrono::duration<double> total_parallel = std::chrono::duration<double>::zero();
            for (int i = 0; i < T; ++i) {
                auto start = std::chrono::steady_clock::now();
                init_omp(a, b, c, M, N);
                matrix_vector_product_omp(a, b, c, M, N);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                total_parallel += elapsed_seconds;
            }
            std::cout << "\tThe parallel calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
            std::cout << "\t" << total_serial / total_parallel << "x acceleration" << std::endl;
            std::cout << "\t-----------------------------------------------------------------------------" << std::endl;
        }
        free(a);
        free(b);
        free(c);
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    }
    return EXIT_SUCCESS;
}

void init_serial(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            a[i * n + j] = i + j;
        c[i] = 0.0;
    }
    for (int j = 0; j < n; ++j)
        b[j] = j;
}

void init_omp(double *a, double *b, double *c, int m, int n) {
    #pragma omp parallel num_threads(NUM_THREADS[thread_idx])
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            for (int j = 0; j < n; ++j)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        }
    }
    for (int j = 0; j < n; ++j)
        b[j] = j;
}

void matrix_vector_product_serial(const double *a, const double *b, double *c, const int m, const int n) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
}

void matrix_vector_product_omp(const double *a, const double *b, double *c, const int m, const int n) {
    #pragma omp parallel num_threads(NUM_THREADS[thread_idx])
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i)
            for (int j = 0; j < n; ++j)
                c[i] += a[i * n + j] * b[j];
    }
}
