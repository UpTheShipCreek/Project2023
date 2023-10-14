#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748


class HFunction{
    std::vector<double> V;
    double T;
    double  W = 100; // I don't get where we get the value for this, need to change it someday
    Random Rand;

    public:
    hFunction(){
        this->V = Rand.generate_vector_normal(DIMENSIONS);
        this->T = Rand.generate_double_uniform(1.0, 2.0); // Changed the min from 0 to 1 and max from 1 to 2, since we were instructed to ensure that p*v + t is positive
    }

    double evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
        double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
        
        double result = (product + this->T)/ this->W;

        return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
    }
};

class FFunction{
    int numOfH;
    std::vector<hFunction*> H;
    std::vector<int> R;
    Random Rand;

    public:
    FFunction(){
        int i;
        for (i=0; i < this-> numOfH; i++){
            hFunction* h = new hFunction();
            (this->H).emplace_back(h);
            (this->R).push_back(Rand.generate_int_uniform(0, 1)); // Generate and save the r value
        }

    }

    ~FFunction(){
        for (hFunction* h: this->H){
            delete h;
        }
    }

};

class HashTable{
    int 
}