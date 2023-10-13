#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748
#define MODULO 15000 
#define LSH_TABLE_SIZE 7500 // 60000/8
#define WINDOW 100 // Test orders of magnitude

// Define a struct to store distance and the corresponding ImageVector
class DistanceImagePair {
    double Distance;
    ImageVector* Image;

    public:
    // Constructor to initialize the values
    DistanceImagePair(double dist, ImageVector* img){
        this->Distance = dist;
        this->Image = img;
    }

    ImageVector* get_image() const{
        return this->Image;
    }

    double get_distance() const{
        return this->Distance;
    }
    
    // Define comparison operators to use with std::priority_queue
    bool operator>(const DistanceImagePair& other) const {
        return Distance > other.Distance;
    }
    bool operator<(const DistanceImagePair& other) const {
        return Distance < other.Distance;
    }
};

class hFunction{
    std::vector<double> V;
    double T;
    double  W = WINDOW;
    Random Rand;

    public:
    hFunction(){
        this->V = Rand.generate_vector_normal(DIMENSIONS); // The N(0,1) is pressuposed in the generate_vector_normal function
        this->T = Rand.generate_double_uniform(0.0, this->W); 
        // So we were explicitly instructed to use the uniform(0,W) distribution for t and N(0,1) for the values of v, 
        // but to also ensure that (p*v + t) is not negative?
        // Why are we allowing negative values in the first place then? 
        // Why are we not shifting the normal distribution to N(1,1) and the uniform distribution to (1, W+1) 
        // so as to not have to worry about negative values?
    }

    double evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
        double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
        
        double result = (product + this->T)/ this->W;

        return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
    }
};

class gFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo, it takes pretty big values, DON'T FORGET TO ASSIGN IT TO THE DEFINE VALUE AFTER THE TESTS ARE OVER
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

// A modified HashTable class that works with the ImageVector class
class HashTable{
    int NumberOfBuckets; // i.e. table size
    gFunction* HashFunction;
    std::unordered_map<int, std::vector<ImageVector*>> Table; // Actual storage

    public:
    HashTable(int num, int k, int m){ // Constructor
        this->NumberOfBuckets = num;
        this->HashFunction = new gFunction(k, m);
    }
    ~HashTable(){ // Destructor
        delete (this->HashFunction);
    }

    void insert(ImageVector* image){ // Insert an image to the correct bucket
        std::vector<double> p = image->get_coordinates(); // Get the coordinates of the image

        int id = (this->HashFunction)->evaluate_point(p); // Get the for the querying trick we saw in the lectures
        image->assign_id(id); // Assign the id to the image

        int bucketId = id % (this->NumberOfBuckets); // Get the bucket id

        if ((this->Table).find(bucketId) == (this->Table).end()){ // Check if the bucket is missing, so as to create it
            (this->Table)[bucketId] = std::vector<ImageVector*>();
        }

        (this->Table)[bucketId].push_back(image); // Add the image to the bucket
    }

    const std::vector<ImageVector*> get_bucket_of_image(ImageVector* image){ // A function that returns a bucket given a certain image
        int bucketId = image->get_id() % (this->NumberOfBuckets); // Get the bucket id
        return (this->Table)[bucketId]; // Fetch the bucket
    }


    std::vector<std::pair<int, double>> get_n_nearest(ImageVector* image, int numberOfNearest){
        printf("%d hashtable::%s Entered\n", __LINE__, __FUNCTION__);
        int number, bucketId;
        int id = image->get_id();
        double distance, dist;
        DistanceImagePair* distpair;

        std::vector<std::pair<int, double>> nearestPairs;
        std::priority_queue<DistanceImagePair, std::vector<DistanceImagePair>, std::greater<DistanceImagePair>> nearest; // Priority Queue to store the nearest neighbors

        if (id == -1){
            return nearestPairs;
        } 
        else{
            bucketId = id % this->NumberOfBuckets;
            std::vector<ImageVector*> bucket = this->Table[bucketId];

            for (int i = 0; i < (int)(bucket.size()); i++){
                if (bucket[i]->get_id() == id && bucket[i] != image){ // Avoid comparing image to itself
                    distance = eucledian_distance(image->get_coordinates(), bucket[i]->get_coordinates());
                    printf("%d hashtable::%s Distance = %f\n", __LINE__, __FUNCTION__, distance);

                    distpair = new DistanceImagePair(distance, bucket[i]);
                    nearest.push(pair);

                    printf("%d hashtable::%s nearest.size = %ld\n", __LINE__, __FUNCTION__, nearest.size());
                    if ((int)(nearest.size()) > numberOfNearest){ // If the number of nearest neighbors is greater than the number of nearest neighbors we want to return
                        nearest.pop(); // Remove the farthest neighbor to maintain "n" nearest neighbors
                    }
                }
            }
            while (!nearest.empty()){
                number = nearest.top().get_image()->get_number();
                dist = nearest.top().get_distance();
                
                nearestPairs.push_back({number, dist});
                nearest.pop();
            }
            return nearestPairs;
        }
    }   
};

class LSH{
    int K, L, M;
    std::vector<HashTable*> Tables;

    public:
    LSH(int l, int k, int modulo){
        printf("%d %s Entered constructor\n", __LINE__, __FUNCTION__);
        this->K = k;
        this->L = l;
        this->M = modulo;
        HashTable* hashtable;

        for (int i = 0 ; i < this-> L; i++){
            hashtable = new HashTable(LSH_TABLE_SIZE, this->K, this->M);
            (this->Tables).push_back(hashtable);
        }
        printf("%d %s Exiting constructor\n", __LINE__, __FUNCTION__);
    }
    ~LSH(){
        printf("%d %s Entered destructor\n", __LINE__, __FUNCTION__);
        for(HashTable* hashtable : this->Tables){
            delete hashtable;
        }
        printf("%d %s Exiting destructor\n", __LINE__, __FUNCTION__);
    }

    void load_data(std::vector<ImageVector> images){ // Load the data to the LSH
        for (int i = 0; i < (int)(images.size()); i++){
            for (int j = 0; j < this->L; j++){
                (this->Tables)[j]->insert(&images[i]);
            }
        }
    }

    std::vector<std::pair<int, double>> get_n_nearest(ImageVector* image, int numberOfNearest){
        printf("%d lsh::%s Entered\n", __LINE__, __FUNCTION__);
        std::vector<std::pair<int, double>> nearestImages;

        int tableNearest = std::ceil(static_cast<double>(numberOfNearest) / this->L);
        printf("%d lsh::%s numberOfNearest:%d/this->L:%d = tableNearest:%d\n", __LINE__, __FUNCTION__, numberOfNearest, this->L,tableNearest);
        //HashTable* table;
        // soFarNearest, needs to work with the reworked get_n_nearest of the hashtable, i.e. it needs to be of type std::vector<std::pair<int, double>>
        std::vector<std::pair<int, double>> nearest;
        std::vector<std::pair<int, double>> newNearest;
        for(auto& table : this->Tables){ // Save the nearest of all tables to a new vector that will contain all, it suffices then to sort this vector and return the first numberOfNearest elements
            newNearest = table->get_n_nearest(image, tableNearest);
            printf("%d lsh::%s newNearest= %ld\n", __LINE__, __FUNCTION__,newNearest.size());
            nearest.insert(nearest.end(), newNearest.begin(), newNearest.end());
        }
        printf("%d %s Saved all the images into one big vector\n", __LINE__, __FUNCTION__);
        // We need to sort using the second value, i.e. the distance
        std::sort(nearest.begin(), nearest.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b){
            return a.second < b.second;
        });
        printf("%d %s Sorted the images\n", __LINE__, __FUNCTION__);
        // We need to return the first numberOfNearest elements
        //nearestImages.insert(nearestImages.end(), nearest.begin(), nearest.begin() + numberOfNearest);
        //printf("%d %s Inserted the first n images into the return vector\n", __LINE__, __FUNCTION__);
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
    int modulo = MODULO;
    int k = 4;
    int l = 5; 
    LSH lsh(l, k, modulo); // The distinct values should be the same number as the modulo for a large enough number of tries

    std::vector<ImageVector> images = read_mnist_images("./in/input.dat"); // Read the images
    lsh.load_data(images); // Load the data to the LSH

    printf("%d Loaded data\n", __LINE__);

    std::vector<std::pair<int,double>> nearest = lsh.get_n_nearest(&images[0], 3); // Get the 3 nearest vectors to the first image

    printf("%ld\n",nearest.size());

    for(int i = 0; i < (int)nearest.size(); i++){
        printf("<%d, %f>\n", nearest[i].first, nearest[i].second);
    }

    return 0;
}