#include <stdio.h>
#include "random_functions.h"


#define DIMENSIONS 2 // the dimension of the space this hyperplane resides

class Hyperplane{
    double Coefficients[DIMENSIONS]; // the A,B,C...D in the hyperplane equation: Ax + By + Cz + ... + Hw + D = 0, which are the dimensions
    double Constant; // the D in the equation above

    public:
    Hyperplane(){
        // The range of the randomized doubles we need to create random hyperplanes
        double min_value = -1.0;
        double max_value = 1.0;

        for (int i = 0; i < DIMENSIONS; i++) {
            this->Coefficients[i] = generate_random_double_uniform(min_value, max_value);
        }
        this->Constant = generate_random_double_uniform(min_value, max_value);
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
            boolean = partition[i].evaluate_point_relative_position(point);
            //printf("The %d relative position is %d\n",i+1, boolean);
            code <<= 1; // shift so that we have some space for the next boolean
            code |= boolean; // save the code of the particular plain
        }
        return code; // the number of codes is a bit of a difficult issue 
    }

};

bool distinct(const std::vector<double>& list, double value) {
    for (double item : list) {
        if (item == value) return false;
    }
    return true;
}

int main(void) {
    double point[2];
    int value;
    int c = 0;
    std::vector<double> list;
    HashFunction hash(3);

    for(int i = 0; i < 20; i++){
        point[0] = generate_random_double_uniform(0.0,1.0);
        point[1] = generate_random_double_uniform(0.0,1.0);

        value = hash.evaluate_point(point);
        if(distinct(list,value)) c++;
        
        list.push_back(value);
        //printf("%d\n",value);
    }
    printf("%d\n", c);
    return 0;

}
