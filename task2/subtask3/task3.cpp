#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

const int N = 16500;
const int T = 10;
int THREAD_NUM;

void M_mult_V(const double* matrix, const double* vector, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i) {
        solution[i] = 0.0;
        for (int j = 0; j < N; ++j)
            solution[i] += matrix[i * N + j] * vector[j];
    }
}

void M_mult_V_2(const double* matrix, const double* vector, double* solution) {
    #pragma omp for
    for (int i = 0; i < N; ++i) {
        solution[i] = 0.0;
        for (int j = 0; j < N; ++j)
            solution[i] += matrix[i * N + j] * vector[j];
    }
}

void V_mult_S(const double* vector, const double scalar, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        solution[i] = vector[i] * scalar;
}

void V_mult_S_2(const double* vector, const double scalar, double* solution) {
    #pragma omp for
    for (int i = 0; i < N; ++i)
        solution[i] = vector[i] * scalar;
}

void V_sub(const double* v1, const double* v2, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        solution[i] = v1[i] - v2[i];
}

void V_sub_2(const double* v1, const double* v2, double* solution) {
    #pragma omp for
    for (int i = 0; i < N; ++i)
        solution[i] = v1[i] - v2[i];
}

double norma(const double* vector) {
    double sum = 0;
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        sum += pow(vector[i], 2);
    sum = sqrt(sum);
    return sum;
}

double norma_2(const double* vector) {
    double sum = 0;
    #pragma omp for
    for (int i = 0; i < N; ++i)
        sum += pow(vector[i], 2);
    sum = sqrt(sum);
    return sum;
}

double get_tau(const double* A, double *y) {
    auto Ay = (double*)calloc(N, sizeof(double));
    double yAy = 0;
    double AyAy = 0;

    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; i++) {
		for (int j=0; j < N; j++)
			Ay[i] += A[i * N + j] * y[j];
        yAy += y[i] * Ay[i];
        AyAy += Ay[i] * Ay[i];
    }

    free(Ay);
    return yAy / AyAy;
}

double get_tau_2(const double* A, double *y) {
    auto Ay = (double*)calloc(N, sizeof(double));
    double yAy = 0;
    double AyAy = 0;

    #pragma omp for
    for (int i = 0; i < N; i++) {
		for (int j=0; j < N; j++)
			Ay[i] += A[i * N + j] * y[j];
        yAy += y[i] * Ay[i];
        AyAy += Ay[i] * Ay[i];
    }

    free(Ay);
    return yAy / AyAy;
}

void solve(const double* A, double* x, const double* b, const double error, const int max_steps=1000) {
    auto intermediate_solution = (double*)malloc(N * sizeof(double));
    double norma_b = norma(b);
    double tau;
    for (int s = 0; s < max_steps; ++s) {
        M_mult_V(A, x, intermediate_solution);
        V_sub(intermediate_solution, b, intermediate_solution);

        if (norma(intermediate_solution) / norma_b < error)
            break;

        tau = get_tau(A, intermediate_solution);
        V_mult_S(intermediate_solution, tau, intermediate_solution);
        V_sub(x, intermediate_solution, x);
    }
    free(intermediate_solution);
}

void solve_2(const double* A, double* x, const double* b, const double error, const int max_steps=100) {
    auto intermediate_solution = (double*)malloc(N * sizeof(double));
    double norma_b = norma_2(b);
    double tau;
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        for (int s = 0; s < max_steps; ++s) {
            M_mult_V_2(A, x, intermediate_solution);
            V_sub_2(intermediate_solution, b, intermediate_solution);

            if (norma_2(intermediate_solution) / norma_b < error)
                break;

            tau = get_tau_2(A, intermediate_solution);
            V_mult_S_2(intermediate_solution, tau, intermediate_solution);
            V_sub_2(x, intermediate_solution, x);
        }
    }
    free(intermediate_solution);
}

void init_all(double* A, double* b, double* answer) {
    // Initialization of the matrix A
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i == j)
                A[i * N + j] = 2.0;
            else
                A[i * N + j] = 1.0;
    // Initialization of the vector b
    for (int i = 0; i < N; ++i)
        b[i] = N + 1;
    // Initialization of the answer vector
    for (int i = 0; i < N; ++i)
        answer[i] = 1.0;
}

double MSE(const double* x, const double* answer) {
    double err = 0.0;
    for (int i = 0; i < N; ++i)
        err += pow(answer[i] - x[i], 2);
    err /= N;
    return err;
}

int main() {
    auto A = (double*)malloc(N * N * sizeof(double));
    auto b = (double*)malloc(N * sizeof(double));
    auto answer = (double*)malloc(N * sizeof(double));
    auto x = (double*)malloc(N * sizeof(double));
    const double error = 0.00001;
    const double lr = 0.0001;

    init_all(A, b, answer);

    std::fstream csv("results.csv", std::ios::out);

    for (THREAD_NUM = 1; THREAD_NUM <= 80; ++THREAD_NUM) {
        std::cout << "\tTHE RESULTS FOR " << THREAD_NUM << " THREADS:" << std::endl;

        std::cout << "THE FIRST VARIANT: " << std::endl;
        std::chrono::duration<double> total_parallel = std::chrono::duration<double>::zero();
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < N; ++j)
                x[j] = 0.0;
            auto start = std::chrono::steady_clock::now();
            solve(A, x, b, error);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_parallel += elapsed_seconds;
            std::cout << elapsed_seconds.count() << std::endl;
            csv << elapsed_seconds.count() << ",";
        }
        std::cout << "\tThe calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
        csv << total_parallel.count() / T << ",,";

        std::cout << "Error = " << MSE(x, answer) << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << x[i] << " ";
        std::cout << std::endl;

        std::cout << "THE SECOND VARIANT: " << std::endl;
        total_parallel = std::chrono::duration<double>::zero();
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < N; ++j)
                x[j] = 0.0;
            auto start = std::chrono::steady_clock::now();
            solve_2(A, x, b, error);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_parallel += elapsed_seconds;
            std::cout << elapsed_seconds.count() << std::endl;
            csv << elapsed_seconds.count() << ",";
        }
        std::cout << "\tThe calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
        csv << total_parallel.count() / T << "\n";

        std::cout << "Error = " << MSE(x, answer) << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << x[i] << " ";
        std::cout << std::endl;
        std::cout << "\t-----------------------------------------------------------------------------" << std::endl;
    }
    csv.close();
    free(A);
    free(b);
    free(answer);
    free(x);
    return 0;
}
