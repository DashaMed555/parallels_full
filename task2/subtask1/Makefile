parallel_test: task1.o
	g++ task1.o -fopenmp -o parallel_test

task1.o: task1.cpp
	g++ -c -fopenmp task1.cpp

clean:
	rm task1.o parallel_test
