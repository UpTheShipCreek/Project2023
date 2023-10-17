#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"
#include "image_util.h"

#define DIMENSIONS 748
#define MODULO INT_MAX - 5
#define LSH_TABLE_SIZE 3750 // 60000/2^n
#define WINDOW 1500 // Test orders of magnitude
#define MEAN 0.0
#define STANDARD_DEVIATION 1.0
#define QUERY_NUMBER 10

class HashFunction{
    public:
    virtual int evaluate_point(std::vector<double> p) = 0;
};

class hFunction{
    std::vector<double> V;
    double T;
    double  W = WINDOW;
    Random Rand;

    public:
    hFunction(){
        this->V = Rand.generate_vector_normal(DIMENSIONS, MEAN, STANDARD_DEVIATION); // The N(0,1) distribution
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

class fFunction : public HashFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo
    std::vector<std::shared_ptr<hFunction>> H;
    std::vector<int> R; // The r values that will be used in the g function
    Random Rand; // Random generator

    public:
    fFunction(int k, int m){
        this->K = k;
        this->M = m;
        for(int i = 0; i < this->K; i++){
            std::shared_ptr<hFunction> h = std::make_shared<hFunction>(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
            (this->H).push_back(h); // Save its pointer to the vector
            (this->R).push_back(Rand.generate_int_uniform(1, 100)); // Generate and save the r value
        }
    }
    int evaluate_point(std::vector<double> p) override{
        int res;
        int sum = 0;
        for(int i = 0; i < this->K; i++){
            sum += (this->R)[i] * ((this->H)[i]->evaluate_point(p)); 
        }
        res = sum % M;

        return res;
    }
};

class gFunction : public HashFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo
    std::vector<std::shared_ptr<hFunction>> H;
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
    int evaluate_point(std::vector<double> p) override{
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

class HashTable{
    int NumberOfBuckets;
    std::shared_ptr<HashFunction> HF;
    std::unordered_map<int, std::vector<std::shared_ptr<ImageVector>>> Table;
    std::unordered_map<int, int> NumberToId; // <number, id> pairs

    public:
    HashTable(int num, std::shared_ptr<HashFunction> hashfunction){ // Constructor
        this->NumberOfBuckets = num;
        this->HF = hashfunction;
    }
    bool same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2){ // Compares the id of two images, used in the querying trick
        return NumberToId[image1->get_number()] == NumberToId[image2->get_number()];
    }
    void insert(std::shared_ptr<ImageVector> image){ // Insert an image to the hash table and save its id
        std::vector<double> p = image->get_coordinates();
        int id = HF->evaluate_point(p);

        NumberToId[image->get_number()] = id;

        int bucketId = id % NumberOfBuckets;

        if (Table.find(bucketId) == Table.end()) {
            Table[bucketId] = std::vector<std::shared_ptr<ImageVector>>();
        }

        Table[bucketId].push_back(image);
    }

   const std::vector<std::shared_ptr<ImageVector>>& get_bucket_of_image(std::shared_ptr<ImageVector> image){ // Returns the bucket a specific image resides in 
        int bucketId = NumberToId[image->get_number()] % NumberOfBuckets;
        return Table[bucketId];
    }
};


class LSH{
    int K, L, M;
    std::vector<std::shared_ptr<HashTable>> Tables;

    public:
    LSH(int l, int k, int modulo, int tableSize){
        printf("Creating LSH... ");
        fflush(stdout);
        this->L = l;
        this->K = k;
        this->M = modulo;

        Tables.reserve(L);

        for (int i = 0; i < L; i++) {
            std::shared_ptr<HashFunction> hashFunction = std::make_shared<gFunction>(this->K, this->M);
            Tables.push_back(std::make_shared<HashTable>(tableSize, hashFunction));
        }

        printf("Done\n");
        fflush(stdout);
    }

    void load_data(std::vector<std::shared_ptr<ImageVector>> images){ // Load the data to the LSH
        printf("Initializing LSH tables... ");
        fflush(stdout);
        for (int i = 0; i < (int)(images.size()); i++){
            for (int j = 0; j < this->L; j++){
                (this->Tables)[j]->insert(images[i]);
            }
        }
        printf("Done\n");
        fflush(stdout);
    }
    std::vector<std::pair<double, int>> approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest){
        int i, j, imageNumber;
        double distance;

        // Ignore itself and every other image it has met before
        std::vector<int> ignore;
        std::vector<int>::iterator it; // Initializing the iteration variable
        ignore.push_back(image->get_number());

        // I will be using a priority queue to keep the k nearest neighbors
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> nearest;

        // Let b ← Null; db ← ∞; initialize k best candidates and distances;
        std::vector<std::pair<double, int>> nearestImages;

        // The pointer of each bucket
        std::vector<std::shared_ptr<ImageVector>> bucket;
        
        // for i from 1 to L do
        for(i = 0; i < this->L; i++){
            bucket = Tables[i]->get_bucket_of_image(image);
            // for each item p in bucket gi (q) do
            for(j = 0; j < (int)(bucket.size()); j++){ 
                // Get the number of the image
                imageNumber = bucket[j]->get_number();

                // See if we have encountered it before
                it = std::find(ignore.begin(), ignore.end(), imageNumber);
                
                if((Tables[i]->same_id(image, bucket[j])) && (it == ignore.end())){ // Query trick + ignore the images we have encountered before
                    // Ignore the image if you find it again
                    ignore.push_back(imageNumber);

                    // dist(p,q)
                    distance = eucledian_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                    
                    // if dist(q, p) < db = k-th best distance then b ← p; db ← dist(q, p), implemented with a priority queue
                    nearest.push(std::make_pair(distance, imageNumber));    

                    if ((int)(nearest.size()) > numberOfNearest){
                        nearest.pop();
                    }  
                }
            }
        }
        // Fill up the a structure that we can return
        while (!nearest.empty()){
            nearestImages.push_back(nearest.top());
            nearest.pop();
        }
        std::vector<std::pair<double,int>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
        return reversed;
    }
    std::vector<std::pair<double, int>> approximate_range_search(std::shared_ptr<ImageVector> image, double r){
        int i, j, imageNumber;
        double distance;

        // Ignore itself and every other image it has met before
        std::vector<int> ignore;
        std::vector<int>::iterator it; // Initializing the iteration variable
        ignore.push_back(image->get_number());

        // The returned vector
        std::vector<std::pair<double, int>> inRangeImages;

        // The pointer of each bucket
        std::vector<std::shared_ptr<ImageVector>> bucket;

        // for i from 1 to L do
        for(i = 0; i < L; i++){
            bucket = Tables[i]->get_bucket_of_image(image);
            // for each item p in bucket gi (q) do
            for(j = 0; j < (int)bucket.size(); j++){
                // Get the number of the image
                imageNumber = bucket[j]->get_number();

                // See if we have encountered it before
                it = std::find(ignore.begin(), ignore.end(), imageNumber);

                if((Tables[i]->same_id(image, bucket[j])) && (it == ignore.end())){ // Query trick + ignore the images we have encountered before
                    // Ignore the image if you find it again
                    ignore.push_back(imageNumber);

                    // if dist(q, p) < r then output p
                    distance = eucledian_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                    if(distance < r){
                        inRangeImages.push_back(std::make_pair(distance, imageNumber));
                    }
                }
                // if large number of retrieved items (e.g. > 20L) then return
                if((int)inRangeImages.size() > 20*(this->L)) return inRangeImages;
            }
        }
        return inRangeImages;
    }
};

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    int k = 4, L = 5, N = 1; // Default values  
    double R = 2000.0;                           
    int opt;
    extern char *optarg; 
    std::string inputFile, queryFile, outputFileName;
    int cmdNecessary = 0;
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //


    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
    while ((opt = getopt(argc, argv, "d:q:k:L:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
        switch (opt) {
            case 'd':                                                               //Files
                inputFile = optarg;
                cmdNecessary++;
                break;
            case 'q':
                queryFile = optarg;
                cmdNecessary++;
                break;
            case 'o':
                outputFileName = optarg;
                cmdNecessary++;
                break;
            case 'k':                                                               //Parameters
                k = atoi(optarg);
                break;
            case 'L':
                L = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'R':
                R = atof(optarg);
                break;
        }
    }

    if(cmdNecessary != 3){
        printf("Program execution requires an input file, output file and a query file.");
        return -1;
    }
    else{
        printf("Program will proceed with values: k = %d, L = %d, N = %d, R = %f\n", k, L, N, R);
    }
    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //
    FILE* outputFile = fopen(outputFileName.c_str(), "w");
    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE LSH -------------------------- //
    // ------------------------------------------------------------------- //
    LSH lsh(L, k, MODULO, LSH_TABLE_SIZE);
    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE LSH -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------- METHOD INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::pair<double, int>> nearest_approx;
    std::vector<std::pair<double, int>> nearest_exhaust;
    std::vector<std::pair<double, int>> range_approx;
    // ------------------------------------------------------------------- //
    // ---------------------- METHOD INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_approx = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto duration_exhaust = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------ OPEN AND LOAD INPUT ---------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images(queryFile, 0);
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, (int)queries.size());
    images.insert(images.end(), queries.begin(), queries.end()); // Merge the two vectors of images so you can load them all at once
    lsh.load_data(images); // Load the data to the LSH
    // ------------------------------------------------------------------- //
    // ------------------------ OPEN AND LOAD INPUT ---------------------- //
    // ------------------------------------------------------------------- //
    int i = 0;
    while(i < QUERY_NUMBER  && i < (int)(queries.size())){
        // ------------------------------------------------------------------- //
        // ----------------------- APPROXIMATE NEAREST ----------------------- //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now(); // Start the timer
        nearest_approx = lsh.approximate_k_nearest_neighbors(queries[i], N); // Get the k approximate nearest vectors to the query
        end  = std::chrono::high_resolution_clock::now(); // End the timer 
        duration_approx = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
        // ------------------------------------------------------------------- //
        // ----------------------- APPROXIMATE NEAREST ----------------------- //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now(); // Start the timer
        nearest_exhaust = exhaustive_nearest_neighbor_search(images, queries[i], N); // Get the k real nearest vectors to the query
        end  = std::chrono::high_resolution_clock::now(); // End the timer 
        duration_exhaust = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
        // ------------------------------------------------------------------- //
        // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ------------------------ APPROXIMATE RANGE ------------------------ //
        // ------------------------------------------------------------------- //
        range_approx = lsh.approximate_range_search(queries[i], R); // Get all the images that are in range R from the query
        // ------------------------------------------------------------------- //
        // ------------------------ APPROXIMATE RANGE ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ---------------------------- WRITES ------------------------------- //
        // ------------------------------------------------------------------- //
        write_approx_exhaust(queries[i], nearest_approx, nearest_exhaust, duration_approx, duration_exhaust, outputFile); 
        write_r_near(queries[i], range_approx, outputFile);
        // ------------------------------------------------------------------- //
        // ---------------------------- WRITES ------------------------------- //
        // ------------------------------------------------------------------- //
        i++;
    }
    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //
    fclose(outputFile);
    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //

    return 0;
}