#include "random_functions.h" // includes <random> and <vector>

std::random_device Random::rd;
std::mt19937 Random::gen(Random::rd());

int Random::generate_int_uniform(const int min, const int max){

    // Might need to change this initialization if it ends up being an important hinderance
    // std::random_device rd; // Initiate a contingency factor
    // std::mt19937 gen(rd()); // And seed them into the Mersenne Twister pseudorandom number generator 

    std::uniform_int_distribution<> distribution(min, max);

    return distribution(gen);
}

double Random::generate_double_uniform(const double min, const double max){

    // std::random_device rd; // Initiate a contingency factor
    // std::mt19937 gen(rd()); // And seed them into the Mersenne Twister pseudorandom number generator 

    std::uniform_real_distribution<> distribution(min, max);

    return distribution(gen);
}

double Random::generate_double_normal(const double mean, const double standardDeviation){
    
    // std::random_device rd; 
    // std::mt19937 gen(rd()); 

    std::normal_distribution<double> distribution(mean, standardDeviation);

    return distribution(gen);
}

std::vector<double> Random::generate_vector_normal(int size, const double mean, const double standardDeviation){ // Size is the dimension of the vector
    std::vector<double> vec;

    for (int i = 0; i < size; i++) { // Fill the vector with values up to its dimensions
        double randomValue = generate_double_normal(mean, standardDeviation); // d-vector ~ N(0,1)^d from the notes
        vec.push_back(randomValue); 
    }

    return vec;
}

std::vector<double> Random::generate_vector_uniform(int size, const double min, const double max){ // Size is the dimension of the vector
    std::vector<double> vec;

    for (int i = 0; i < size; i++) { // Fill the vector with values up to its dimensions
        double randomValue = generate_double_uniform(min, max); 
        vec.push_back(randomValue); 
    }

    return vec;
}
