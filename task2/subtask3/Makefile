option = -fopenmp

parallel_test: task3.o
	g++-13 task3.o ${option} -o $@

task3.o: task3.cpp
	g++-13 -c ${option} task3.cpp

clean:
	rm task3.o parallel_test
