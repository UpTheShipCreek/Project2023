#include <stdio.h>
#include <vector>
#include <cmath>

#include "random_functions.h"

#define DIMENSIONS 748


class hFunction{
    std::vector<double> V;
    double T;
    double  W = 1.0; // I don't get where we get the value for this, need to change it someday

    public:
    hFunction(){
        this->V = generate_random_vector_normal(DIMENSIONS);
        this->T = generate_random_double_uniform(0.0, 1.0);
    }

    double evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
        double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
        
        double result = (product + this->T)/ this->W;

        return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
    }
};

class gFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo 
    std::vector<hFunction*> H; // Since I am saving pointers I'll need to remove those functions

    public:
    gFunction(int k, int m){
        this->K = k;
        this->M = m;
        for(int i = 0; i < this->K; i++){
            hFunction* h = new hFunction(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
            (this->H).push_back(h); // Save its pointer to the vector
        }
    }

    ~gFunction() {
        for (hFunction* h : this->H) {
            delete h; // Delete each dynamically allocated hFunction
        }
    }

    int evaluate_point(std::vector<double> p){
        int r, res;
        int sum = 0;
        for(int i = 0; i < this->K; i++){
            r = generate_random_int_uniform(1, 9);
            sum += r * H[i]->evaluate_point(p);
        }
        res = sum % M;
        if(res >= 0) return res;
        else return (M + res); // Invoking the modular negation property in order to keep the result positive
    }
};

class HashTable{
    int NumberOfBuckets;
    gFunction* HashFunction;
    std::unordered_map<int, std::vector<std::vector<double>>> Table; // Actual storage

    HashTable(int num, int k, int m){ // Constructor
        this->NumberOfBuckets = num;
        this->HashFunction = new gFunction(k, m);
    }
    ~HashTable(){ // Destructor
        delete (this->HashFunction);
    }

    void insert(std::vector<double> p){
        int bucketId = (this->HashFunction)->evaluate_point(p); // Get the proper bucket id


        if ((this->Table).find(bucketId) == (this->Table).end()){ // Check if the bucket is missing, so as to create it
            (this->Table)[bucketId] = std::vector<std::vector<double>>();
        }
        
        (this->Table)[bucketId].push_back(dataPoint); // Add the data point to the bucket
    }

    
};

bool distinct(const std::vector<double>& list, double value) {
    for (double item : list) {
        if (item == value) return false;
    }
    return true;
}

int main(void){
    int c = 0;
    std::vector<double> list;
    int modulo = 10;
    gFunction g(4,modulo); // The distinct values should be the same number as the modulo for a large enough number of tries
    int bucket_id;

    for(int i = 0; i < 10000; i++){
        bucket_id = g.evaluate_point(generate_random_vector_normal(DIMENSIONS));
        if(distinct(list,bucket_id)) c++;
        list.push_back(bucket_id);
    }
    printf("%d distinct values\n", c); 
    return 0;
}