N = 512
accuracy = 0.000001
iterations_num = 1000000

BOOST_LIB = lboost_program_options
COMPILER = pgc++
DEVICE = gpu
# gpu || host || multicore

default: result.dat

result.dat: solver
	CUDA_VISIBLE_DEVICES="2" ./solver --mesh_size $(N) --accuracy $(accuracy) --iterations_num $(iterations_num)

solver: therm.cpp
	$(COMPILER) -std=c++11 therm.cpp -o solver -lm -$(BOOST_LIB) -acc=$(DEVICE) -Minfo=all -fast

clean:
	rm -f solver result.dat
