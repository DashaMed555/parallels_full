option = -std=c++20
compiler = g++

parallel_test: main.o
	${compiler} $^ ${option} -o $@

main.o: main.cpp
	${compiler} -c ${option} $^

clean:
	rm main.o parallel_test
