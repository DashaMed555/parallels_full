parallel_test: task2.o
	g++-13 task2.o -fopenmp -o parallel_test

task2.o: task2.cpp
	g++-13 -c -fopenmp task2.cpp

clean:
	rm task2.o parallel_test
