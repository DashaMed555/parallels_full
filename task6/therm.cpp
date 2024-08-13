#include <boost/program_options.hpp>

namespace po = boost::program_options;

// PARAMETERS SECTION

#define IDX(i, j) (i) * N + (j)

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

#include <iostream>
#include <chrono>

void init_grid(double* grid) {
    for (int i = 0; i < N; ++i) {
        grid[IDX(0, i)] = MinYMinX + i * (MinYMaxX - MinYMinX) / (N - 1);
        grid[IDX(N - 1, i)] = MaxYMinX + i * (MaxYMaxX - MaxYMinX) / (N - 1);
        grid[IDX(i, 0)] = MinYMinX + i * (MaxYMinX - MinYMinX) / (N - 1);
        grid[IDX(i, N - 1)] = MinYMaxX + i * (MaxYMaxX - MinYMaxX) / (N - 1);
    }
}

void Jacobi_method(double* grid) {
    auto new_grid = (double*)malloc(N * N * sizeof(double));
    init_grid(new_grid);
    double* p;
    double maxdiff;
    int step;

    #pragma acc data copy(grid[0:N*N]) copyin(new_grid[0:N*N]) 
    {
    for (step = 0; step < max_steps; ++step) {
        maxdiff = 0;
        #pragma acc parallel loop reduction(max:maxdiff) present(grid[0:N*N], new_grid[0:N*N])
        for (int i = 1; i < N - 1; ++i) 
            #pragma acc loop
            for (int j = 1; j < N - 1; ++j) {
                new_grid[IDX(i, j)] = (grid[IDX(i - 1, j)] + grid[IDX(i + 1, j)] + grid[IDX(i, j - 1)] + grid[IDX(i, j + 1)]) * 0.25;
                maxdiff = fmax(maxdiff, fabs(grid[IDX(i, j)] - new_grid[IDX(i, j)]));
            }

        p = grid;
        grid = new_grid;
        new_grid = p;

        if (maxdiff < EPS)
            break;

        printf("%lf >= %lf\r", maxdiff, EPS);
        fflush(stdout);
    }
    }

    printf("\33[2K\r");
    printf("Error: %f\n", maxdiff);
    printf("Iterations num: %d\n", step);

	fflush(stdout);
    #pragma acc exit data delete(new_grid[0:N*N])
    free(new_grid);
}

int main(int argc, char* argv[]) {
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

    double* grid = (double*)malloc(N * N * sizeof(double));
    init_grid(grid);

	auto start = std::chrono::steady_clock::now();
	Jacobi_method(grid);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "It finished after " << elapsed_seconds.count() << " seconds!" << std::endl;
	
    FILE* out_file = fopen(OUT_FILE, "wb");
	fwrite(grid, sizeof(double), N * N, out_file);
	fclose(out_file);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++)
    //         std::cout << grid[IDX(i, j)] << " ";
    //     std::cout << std:: endl;
    // }

    free(grid);
    return 0;
}
