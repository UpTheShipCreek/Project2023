#ifndef RANDOM_FUNCTIONS_H //header guard for the vector header file which I know will be included elsewhere
#define RANDOM_FUNCTIONS_H

#include <random>
#include <vector>

int generate_random_int_uniform(const int min, const int max);
double generate_random_double_uniform(const double min, const double max);
double generate_random_double_normal(const double mean, const double standardDeviation);

std::vector<double> generate_random_vector_normal(int size);
std::vector<double> generate_random_vector_uniform(int size);

#endif