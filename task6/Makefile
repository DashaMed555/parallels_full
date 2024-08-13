N = 1024
accuracy = 0.000001
iterations_num = 1000000

BOOST_LIB = lboost_program_options
COMPILER = g++

default: result.dat

default: show_image

.PHONY:
show_image: result.dat
	gnuplot -e "plot 'result.dat' binary \
		array=($(N), $(N)) format='%lf' with image; pause -1"

result.dat: solver
	./solver --mesh_size $(N) --accuracy $(accuracy) --iterations_num $(iterations_num)

solver: therm.cpp
	$(COMPILER) -std=c++11 therm.cpp -o solver -lm -$(BOOST_LIB)

clean:
	rm -f solver result.dat
