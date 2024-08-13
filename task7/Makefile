N = 10
accuracy = 0.000001
iterations_num = 1000000

BOOST_LIB = lboost_program_options
CUDA_INCLUDE = /opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include
COMPILER = pgc++
DEVICE = gpu
# gpu || host || multicore

default: result.dat

result.dat: solver
	CUDA_VISIBLE_DEVICES="2" ./solver --mesh_size $(N) --accuracy $(accuracy) --iterations_num $(iterations_num)

solver: therm.cpp
	$(COMPILER) -std=c++11 therm.cpp -o solver -lm -$(BOOST_LIB) -I$(CUDA_INCLUDE) -acc=$(DEVICE) -Minfo=all -fast -cudalib=cublas

clean:
	rm -f solver result.dat
