#include <stdio.h>
#include <vector>
#include <random>

#define DIMENSIONS 2 // the dimension of the space this hyperplane resides

double random_double(double min, double max){
    // Create a random number generator engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister PRNG

    // Create a distribution for generating random doubles in the specified range
    std::uniform_real_distribution<double> dist(min, max);

    return dist(gen);
}

class Hyperplane{
    double Coefficients[DIMENSIONS]; // the A,B,C...D in the hyperplane equation: Ax + By + Cz + ... + Hw + D = 0, which are the dimensions
    double Constant; // the D in the equation above

    public:
    Hyperplane(){
        // The range of the randomized doubles we need to create random hyperplanes
        double min_value = -1.0;
        double max_value = 1.0;

        for (int i = 0; i < DIMENSIONS; i++) {
            this->Coefficients[i] = random_double(min_value, max_value);
        }
        this->Constant = random_double(min_value, max_value);
    }

    Hyperplane(double* coefficients, double constant){
        for (int i = 0; i < DIMENSIONS; i++){
            this->Coefficients[i] = coefficients[i];
        }
        this->Constant = constant;
    }
    // seems like int is easier to work with
    int evaluate_point_relative_position(double* point){ // If the point is above the hyperplane then returns True
        double result = 0.0;
        for (int i = 0; i < DIMENSIONS; i++){ // putting the numbers of the point in the hyperplane formula
            result += point[i] * this->Coefficients[i]; 
        }
        result += this->Constant; // Add the last element as the bias term

        return result > 0.0;
    }
};

class HashFunction{
    int Number_of_Partitions;
    std::vector<Hyperplane> partition; // Store partitions here

    public:
    HashFunction(int number){
        Number_of_Partitions = number;

        // Create partitions
        for (int i = 0; i < number; i++){
            Hyperplane hyperplane;
            partition.push_back(hyperplane);
        }
    }
    // find the unique code of the point, we are working with binary operators and so the results are of type 1010011... of Number_of_Partitions length
    int evaluate_point(double* point){
        int code = 0;
        int boolean;
        for (int i = 0; i < Number_of_Partitions; i++) { // for each partition see if the point is above or below
            bool boolean = partition[i].evaluate_point_relative_position(point);
            printf("The %d relative position is %d\n",i+1, boolean);
            code <<= 1; // shift so that we have some space for the next boolean
            code |= boolean; // save the code of the particular plain
        }
        return code;
    }

};

int main(void) {
    double point[2];

    for(int i = 0; i < 10; i++){
        point[0] = random_double(0.0,1.0);
        point[1] = random_double(0.0,1.0);
        HashFunction hash(5);
        printf("%d\n",hash.evaluate_point(point));
    }
    return 0;

}
