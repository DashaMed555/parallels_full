#include <boost/program_options.hpp>

namespace po = boost::program_options;

// PARAMETERS SECTION

#define MinYMinX 10.
#define MinYMaxX 20.
#define MaxYMaxX 30.
#define MaxYMinX 20.

// output file name
#define OUT_FILE "result.dat"

// mesh size
int N;

// parameters
double EPS;
int max_steps;

// END OF PARAMETERS SECTION

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <chrono>

// vector size
int SIZE;

// get matrix value (SIZE x SIZE)
double get_a(int row, int col) {
    if (row + N == col) return 1;
	if (row - N == col) return 1;
	if ((row % N == 0) && ((int(row / N) * N) <= col) && (col < (int(row / N + 1) * N))) return MinYMinX + (col % N) * (MinYMaxX - MinYMinX) / (N - 1);
    if ((row % N == (N - 1)) && ((int(row / N) * N) <= col) && (col < (int(row / N + 1) * N))) return MaxYMinX + (col % N) * (MaxYMaxX - MaxYMinX) / (N - 1);
    if ((col % N == 0) && ((int(row / N) * N) <= col) && (col < (int(row / N + 1) * N))) return MinYMinX + (row % N) * (MaxYMinX - MinYMinX) / (N - 1);
    if ((col % N == (N - 1)) && ((int(row / N) * N) <= col) && (col < (int(row / N + 1) * N))) return MinYMaxX + (row % N) * (MaxYMaxX - MinYMaxX) / (N - 1);
	return 0;
}

void init_b(double* b) {
    for (int i = 0; i < SIZE; ++i)
        if ((double)std::rand() / RAND_MAX < 0.1)
            b[i] = ((double)std::rand() / RAND_MAX * 2 - 1) * 50;
        else
            b[i] = 0;
}

double norma(const double* vector) {
    double sum = 0;
    #pragma acc parallel loop reduction(+ : sum)
    for (int i = 0; i < SIZE; ++i)
        sum += vector[i] * vector[i];
    return sqrt(sum);
}

void A_mult_X_sub_B(const double* x, const double* b, double* solution) {
    #pragma acc parallel loop
    for (int i = 0; i < SIZE; ++i) {
        solution[i] = -b[i];
        for (int j = 0; j < SIZE; ++j)
            solution[i] += get_a(i, j) * x[j];
    }
}

void X_sub_V_mult_S(double* x, const double* v, const double scalar) {
    #pragma acc parallel loop
    for (int i = 0; i < SIZE; ++i)
        x[i] -= v[i] * scalar;
}

double get_tau(const double* y, double* Ay) {
    double yAy = 0;
    double AyAy = 0;

    #pragma acc parallel loop reduction(+ : yAy, AyAy)
    for (int i = 0; i < SIZE; i++) {
        Ay[i] = 0;
		for (int j = 0; j < SIZE; j++)
			Ay[i] += get_a(i, j) * y[j];
        yAy += y[i] * Ay[i];
        AyAy += Ay[i] * Ay[i];
    }
    return yAy / AyAy;
}

void solve(double* x, const double* b, const double error, const int max_steps=100) {
    auto intermediate_solution = (double*)malloc(SIZE * sizeof(double));
    auto temp = (double*)malloc(SIZE * sizeof(double));
    double norm_Axmb;
    double norma_b = norma(b);
    double tau;
    int s;

    #pragma acc data copy(x[0:SIZE]) copyin(b[0:SIZE]) create(intermediate_solution[0:SIZE], temp[0:SIZE])
    {
    for (s = 0; s < max_steps; ++s) {
        A_mult_X_sub_B(x, b, intermediate_solution);

        norm_Axmb = norma(intermediate_solution);
        if (norm_Axmb / norma_b < error)
            break;
        printf("%lf >= %lf\r", norm_Axmb/norma_b, EPS);
        fflush(stdout);

        tau = get_tau(intermediate_solution, temp);
        X_sub_V_mult_S(x, intermediate_solution, tau);
    }
    }

    printf("\33[2K\r");
    printf("Error: %f\n", norm_Axmb/norma_b);
    printf("Iterations num: %d\n", s);
	fflush(stdout);

    free(intermediate_solution);
    free(temp);
}


int main(int argc, char* argv[]) {
    std::srand(888);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("accuracy", po::value<double>(), "set accuracy")
        ("mesh_size", po::value<int>(), "set mesh size")
        ("iterations_num", po::value<int>(), "set iterations num")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("mesh_size")) {
        N = vm["mesh_size"].as<int>();
    } else {
        N = 128;
    }
    SIZE = N * N;

    if (vm.count("accuracy")) {
        EPS = vm["accuracy"].as<double>();
    } else {
        EPS = 0.000001;
    }

    if (vm.count("iterations_num")) {
        max_steps = vm["iterations_num"].as<int>();
    } else {
        max_steps = 1000000;
    }

    auto x = (double*)calloc(SIZE, sizeof(double));
    double* b = (double*)malloc(SIZE * sizeof(double));
    init_b(b);

	auto start = std::chrono::steady_clock::now();
	solve(x, b, EPS, max_steps);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "It finished after " << elapsed_seconds.count() << " seconds!" << std::endl;
	
	std::ofstream out_file(OUT_FILE, std::ios::binary);
	out_file.write(reinterpret_cast<char*>(x), sizeof(double) * SIZE);
	out_file.close();
    
    free(x);
    free(b);
    return 0;
}
