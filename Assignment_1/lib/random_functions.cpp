#include "random_functions.h"

double generate_random_double_uniform(const double min, const double max){

    std::random_device rd; // Initiate a contingency factor
    std::mt19937 gen(rd()); // And seed them into the Mersenne Twister pseudorandom number generator 

    std::uniform_real_distribution<> distribution(min, max);

    return distribution(gen);
}

double generate_random_double_normal(const double mean, const double standardDeviation){
    
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    std::normal_distribution<double> distribution(mean, standardDeviation);

    return distribution(gen);
}

std::vector<double> generate_random_vector_normal(int size){ // Size is the dimension of the vector
    std::vector<double> vec;

    for (int i = 0; i < size; i++) { // Fill the vector with values up to its dimensions
        double randomValue = generate_random_double_normal(0.0, 1.0); // d-vector ~ N(0,1)^d from the notes
        vec.push_back(randomValue); 
    }

    return vec;
}

std::vector<double> generate_random_vector_uniform(int size){ // Size is the dimension of the vector
    std::vector<double> vec;

    for (int i = 0; i < size; i++) { // Fill the vector with values up to its dimensions
        double randomValue = generate_random_double_uniform(0.0, 1.0); 
        vec.push_back(randomValue); 
    }

    return vec;
}
