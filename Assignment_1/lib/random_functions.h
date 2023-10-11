#ifndef RANDOM_FUNCTIONS_H //header guard for the vector header file which I know will be included elsewhere
#define RANDOM_FUNCTIONS_H

#include <random>
#include <vector>

class Random {
    static std::random_device rd; // Static in order not to be initialized again and again
    static std::mt19937 gen;

public:
    int generate_int_uniform(const int min, const int max);
    double generate_double_uniform(const double min, const double max);
    double generate_double_normal(const double mean, const double standardDeviation);
    std::vector<double> generate_vector_normal(int size);
    std::vector<double> generate_vector_uniform(int size);
};


#endif