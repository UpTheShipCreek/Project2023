#include <stdio.h>
#include <unordered_map>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748

class hFunction{
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

class gFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo, it takes pretty big values
    std::vector<hFunction*> H; // Since I am saving pointers I'll need to remove those functions
    std::vector<int> R; // The r values that will be used in the g function
    Random Rand; // Random generator

    public:
    gFunction(int k, int m){
        this->K = k;
        this->M = m;
        for(int i = 0; i < this->K; i++){
            hFunction* h = new hFunction(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
            (this->H).push_back(h); // Save its pointer to the vector
            (this->R).push_back(Rand.generate_int_uniform(1, std::numeric_limits<int>::max())); // Generate and save the r value
        }
    }

    ~gFunction() {
        for (hFunction* h : this->H) {
            delete h; // Delete each dynamically allocated hFunction
        }
    }

    int evaluate_point(std::vector<double> p){
        int res;
        int sum = 0;
        for(int i = 0; i < this->K; i++){
            sum += (this->R)[i] * (this->H)[i]->evaluate_point(p); 
        }
        res = sum % M;
        if(res >= 0) return res;
        else return (M + res); // Invoking the modular negation property in order to keep the result positive, this should be a very rare case after the changes in the p*v + t calculations 
    }
};

class HashTable{
    int NumberOfBuckets;
    gFunction* HashFunction;
    std::unordered_map<int, std::vector<std::vector<double>>> Table; // Actual storage

    public:
    HashTable(int num, int k, int m){ // Constructor
        this->NumberOfBuckets = num;
        this->HashFunction = new gFunction(k, m);
    }
    ~HashTable(){ // Destructor
        delete (this->HashFunction);
    }

    void insert(std::vector<double> p){
        int id = (this->HashFunction)->evaluate_point(p); // Get the for the querying trick we saw in the lectures
        int bucketId = id % (this->NumberOfBuckets); // Get the bucket id


        if ((this->Table).find(bucketId) == (this->Table).end()){ // Check if the bucket is missing, so as to create it
            (this->Table)[bucketId] = std::vector<std::vector<double>>();
        }
        
        p.push_back(id); // Add the id to the end of the point vector. p[784] is the id of the point
        (this->Table)[bucketId].push_back(p); // Add the data point to the bucket
    }

    const std::vector<std::vector<double>> get_bucket_of_point(std::vector<double> p){ // A function that returns a bucket given a certain point
        int bucketId = ((this->HashFunction)->evaluate_point(p)) % (this->NumberOfBuckets); // Get the bucket id
        return (this->Table)[bucketId];
    }

    std::vector<std::vector<double>> get_n_nearest(std::vector<double> p, int numberOfNearest){ // A function that returns the N nearest points of a given point
        int i, maxIndex;
        double maxDistance, distance, dist; 
        std::vector<std::vector<double>> nearest;

        int id = (this->HashFunction)->evaluate_point(p); // Get the id of the point for the querying trick

        int bucketId = id % (this->NumberOfBuckets); // Find its bucket 
        std::vector<std::vector<double>> bucket = (this->Table)[bucketId]; // and fetch it

        for (i = 0; i < (int)bucket.size(); i++){ // For every element in the bucket
            if (bucket[i][784] == id){  // Do compare only those with the same Id
                distance = eucledian_distance(p, bucket[i]);
                if (distance > 0) { // Exclude oneself
                    if ((int)nearest.size() < numberOfNearest){ // If the nearest array is not full, add the vector
                        nearest.push_back(bucket[i]);
                    } 
                    else { // If the nearest array is full
                        maxDistance = 0;
                        maxIndex = 0;
                        for (int j = 0; j < numberOfNearest; j++){ // Find the furthest of the nearest vectors
                            dist = eucledian_distance(p, nearest[j]);
                            if (dist > maxDistance) {
                                maxDistance = dist;
                                maxIndex = j;
                            }
                        }
                        if (distance < maxDistance){ // And compare the furthest with the current vector
                            nearest[maxIndex] = bucket[i];
                        }
                    }
                }
            }
        }
        return nearest;
    }   
};


bool distinct(const std::vector<double>& list, double value) {
    for (double item : list) {
        if (item == value) return false;
    }
    return true;
}

int main(void){
    Random rand;
    int modulo = 10;
    int k = 4;
    int buckets = 10;
    HashTable hashtable(buckets, k, modulo); // The distinct values should be the same number as the modulo for a large enough number of tries

    std::vector<double> point = rand.generate_vector_normal(DIMENSIONS);
    hashtable.insert(point);

    for(int i = 0; i < 10000; i++){
        hashtable.insert(rand.generate_vector_normal(DIMENSIONS));
    }

    std::vector<std::vector<double>> nearest = hashtable.get_n_nearest(point, 3);
    return 0;
}