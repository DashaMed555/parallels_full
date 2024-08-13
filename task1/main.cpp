#if defined(FLOAT)
    using TYPE = float;
#elif defined(DOUBLE)
    using TYPE = double;
#else
    using TYPE = int;
#endif

#include <iostream>
#include <cmath>

int main() {
    const int N = 10000000;
    TYPE* array = (TYPE*)malloc(N * sizeof(TYPE));
    TYPE sum = 0;
    for (int i = 0; i < N; ++i) {
        array[i] = sin(2 * M_PI * i / N);
        sum += array[i];
    }
    std::cout << "The sum is " << sum << "!" << std::endl;
    free(array);
    return 0;
}
