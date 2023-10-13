#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
//#include <memory>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748
#define MODULO 60000
#define LSH_TABLE_SIZE 3750 // 60000/8
#define WINDOW 1000 // Test orders of magnitude

class hFunction{
    std::vector<double> V;
    double T;
    double  W = WINDOW;
    Random Rand;

    public:
    hFunction(){
        this->V = Rand.generate_vector_normal(DIMENSIONS); // The N(0,1) is pressuposed in the generate_vector_normal function
        this->T = Rand.generate_double_uniform(1.0, this->W); 
        // So we were explicitly instructed to use the uniform(0,W) distribution for t and N(0,1) for the values of v, 
        // but to also ensure that (p*v + t) is not negative?
        // Why are we allowing negative values in the first place then? 
        // Why are we not shifting the normal distribution to N(1,1) and the uniform distribution to (1, W+1) 
        // so as to not have to worry about negative values?
    }

    double evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
        double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
        
        double result = (product + this->T)/ this->W;

        //printf("%d hFunction::%s result = %f\n", __LINE__, __FUNCTION__, result);
        return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
    }
};

class gFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo, it takes pretty big values, DON'T FORGET TO ASSIGN IT TO THE DEFINE VALUE AFTER THE TESTS ARE OVER
    std::vector<std::shared_ptr<hFunction>> H; // Since I am saving pointers I'll need to remove those functions
    std::vector<int> R; // The r values that will be used in the g function
    Random Rand; // Random generator

    public:
    gFunction(int k, int m){
        this->K = k;
        this->M = m;
        for(int i = 0; i < this->K; i++){
            std::shared_ptr<hFunction> h = std::make_shared<hFunction>(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
            (this->H).push_back(h); // Save its pointer to the vector
            (this->R).push_back(Rand.generate_int_uniform(1, 100)); // Generate and save the r value
        }
    }

    // ~gFunction() {
    //     for (hFunction* h : this->H) {
    //         delete h; // Delete each dynamically allocated hFunction
    //     }
    // }

    int evaluate_point(std::vector<double> p){
        int res;
        int sum = 0;
        for(int i = 0; i < this->K; i++){
            sum += (this->R)[i] * ((this->H)[i]->evaluate_point(p)); 
        }
        res = sum % M;

        if(res >= 0){
            //printf("%d gFunction::%s result = %d\n", __LINE__, __FUNCTION__, res);
            return res;
        }
        else{
            //printf("%d gFunction::%s result:%d + M:%d = %d\n", __LINE__, __FUNCTION__, res, M, res+M);
            return res+M; // Invoking the modular negation property in order to keep the result positive, this should be a very rare case after the changes in the p*v + t calculations 
        }
    }
};

// A modified HashTable class that works with the ImageVector class
class HashTable{
    int NumberOfBuckets;
    std::shared_ptr<gFunction> HashFunction;
    std::unordered_map<int, std::vector<std::shared_ptr<ImageVector>>> Table;
    std::unordered_map<int, int> NumberToId;

    public:
    HashTable(int num, int k, int m){ // Constructor
        this->NumberOfBuckets = num;
        this->HashFunction = std::make_shared<gFunction>(k, m);
    }
    // ~HashTable(){
    //     delete (this->HashFunction);
    //     for (auto& bucket : this->Table) {
    //         bucket.second.clear(); 
    //     }
    // }

    void insert(std::shared_ptr<ImageVector> image) {
        std::vector<double> p = image->get_coordinates();
        int id = HashFunction->evaluate_point(p);

        NumberToId[image->get_number()] = id;

        int bucketId = id % NumberOfBuckets;

        if (Table.find(bucketId) == Table.end()) {
            Table[bucketId] = std::vector<std::shared_ptr<ImageVector>>();
        }

        Table[bucketId].push_back(image);
    }

    const std::vector<std::shared_ptr<ImageVector>>& get_bucket_of_image(std::shared_ptr<ImageVector> image) {
        int bucketId = NumberToId[image->get_number()] % NumberOfBuckets;

        if (Table.find(bucketId) != Table.end()) {
            return Table[bucketId];
        }

        // Handle the case where the bucket is not found, e.g., return an empty vector or throw an exception.
    }

    std::vector<std::pair<double, int>> get_n_nearest(std::shared_ptr<ImageVector> image, int numberOfNearest) {
        int i, bucketId;
        double distance;
        int id = NumberToId[image->get_number()];
        
        std::vector<std::pair<double, int>> nearestPairs;
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> nearest;

        bucketId = id % NumberOfBuckets;

        std::vector<std::shared_ptr<ImageVector>>& bucket = Table[bucketId];

        for (i = 0; i < static_cast<int>(bucket.size()); i++) {
            if (NumberToId[bucket[i]->get_number()] == id && bucket[i] != image) {
                distance = eucledian_distance(image->get_coordinates(), bucket[i]->get_coordinates());
                nearest.push(std::make_pair(distance, bucket[i]->get_number()));

                if (static_cast<int>(nearest.size()) > numberOfNearest) {
                    nearest.pop();
                }
            }
        }

        while (!nearest.empty()) {
            nearestPairs.push_back(nearest.top());
            nearest.pop();
        }

        return nearestPairs;
    }
};

class LSH {
    int K, L, M;
    std::vector<std::shared_ptr<HashTable>> Tables;

    public:
    LSH(int l, int k, int modulo, int tableSize) : K(k), L(l), M(modulo) {
        Tables.reserve(L);
        for (int i = 0; i < L; i++) {
            Tables.push_back(std::make_shared<HashTable>(tableSize, K, M));
        }
    }

    void load_data(std::vector<std::shared_ptr<ImageVector>> images){ // Load the data to the LSH
        for (int i = 0; i < (int)(images.size()); i++){
            for (int j = 0; j < this->L; j++){
                (this->Tables)[j]->insert(images[i]);
            }
        }
    }

    std::vector<std::pair<double, int>> get_n_nearest(std::shared_ptr<ImageVector> image, int numberOfNearest){
        std::vector<std::pair<double, int>> nearestImages;

        for (int i = 0; i < L; i++) {
            if (Tables[i]) {
                auto newNearest = Tables[i]->get_n_nearest(image, numberOfNearest);
                nearestImages.insert(nearestImages.end(), newNearest.begin(), newNearest.end());
            }
        }

        std::sort(nearestImages.begin(), nearestImages.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });

        nearestImages.resize(std::min(numberOfNearest, static_cast<int>(nearestImages.size())));

        return nearestImages;
    }
};


int main(void){
    int k = 4;
    int l = 5; 
    LSH lsh(l, k, MODULO, LSH_TABLE_SIZE); // The distinct values should be the same number as the modulo for a large enough number of tries

    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("./in/input.dat"); // Read the images
    lsh.load_data(images); // Load the data to the LSH

    //printf("%d Loaded data\n", __LINE__);

    std::vector<std::pair<double, int>> nearest = lsh.get_n_nearest(images[0], 10); // Get the 3 nearest vectors to the first image

    //printf("%ld\n",nearest.size());

    for(int i = 0; i < (int)nearest.size(); i++){
        printf("<%f, %d>\n", nearest[i].first, nearest[i].second);
    }

    return 0;
}