#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748
#define MODULO INT_MAX - 5
#define LSH_TABLE_SIZE 3750 // 60000/2^n
#define WINDOW 2000 // Test orders of magnitude
#define MEAN 0.0
#define STANDARD_DEVIATION 1.0

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
    std::unordered_map<int, int> NumberToId; // <number, id> pairs

    public:
    HashTable(int num, int k, int m){ // Constructor
        this->NumberOfBuckets = num;
        this->HashFunction = std::make_shared<gFunction>(k, m);
    }

    bool same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2){ // Compares the id of two images, used in the querying trick
        return NumberToId[image1->get_number()] == NumberToId[image2->get_number()];
    }

    void insert(std::shared_ptr<ImageVector> image){ // Insert an image to the hash table and save its id
        std::vector<double> p = image->get_coordinates();
        int id = HashFunction->evaluate_point(p);

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
        this->L = l;
        this->K = k;
        this->M = modulo;

        Tables.reserve(L); // Supposed to be making it faster, couldn't tell the difference (guessing the bottleneck is elsewhere)
        for (int i = 0; i < L; i++) {
            Tables.push_back(std::make_shared<HashTable>(tableSize, K, M));
        }
    }

    void load_data(std::vector<std::shared_ptr<ImageVector>> images){ // Load the data to the LSH
        printf("Initializing LSH tables... ");
        for (int i = 0; i < (int)(images.size()); i++){
            for (int j = 0; j < this->L; j++){
                (this->Tables)[j]->insert(images[i]);
            }
        }
        printf("Done\n");
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
        int i, j;
        double distance;

        // The returned vector
        std::vector<std::pair<double, int>> inRangeImages;

        // The pointer of each bucket
        std::vector<std::shared_ptr<ImageVector>> bucket;

        // for i from 1 to L do
        for(i = 0; i < L; i++){
            bucket = Tables[i]->get_bucket_of_image(image);
            // for each item p in bucket gi (q) do
            for(j = 0; j < (int)bucket.size(); j++){
                if(bucket[j]->get_number() != image->get_number()){ // Ignore comparing to itself

                    // if dist(q, p) < r then output p
                    distance = eucledian_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                    if(distance < r){
                        inRangeImages.push_back(std::make_pair(distance, bucket[j]->get_number()));
                    }
                }
                // if large number of retrieved items (e.g. > 20L) then return
                if((int)inRangeImages.size() > 20*(this->L)) return inRangeImages;
            }
        }
        return inRangeImages;
    }
};


int main(void){
    int k = 4;
    int l = 5; 
    LSH lsh(l, k, MODULO, LSH_TABLE_SIZE); // The distinct values should be the same number as the modulo for a large enough number of tries

    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("./in/input.dat"); // Read the images
    lsh.load_data(images); // Load the data to the LSH

    //printf("%d Loaded data\n", __LINE__);

    std::vector<std::pair<double, int>> nearest = lsh.approximate_k_nearest_neighbors(images[0], 10); // Get the k nearest vectors to the first image
    for(int i = 0; i < (int)nearest.size(); i++){
        printf("<%f, %d>\n", nearest[i].first, nearest[i].second);
    }
    printf("\n--------------------------------\n");

    std::vector<std::pair<double, int>> exh = exhaustive_nearest_neighbor_search(images, images[0], 10); // Get the k nearest vectors to the first image
    for(int i = 0; i < (int)exh.size(); i++){
        printf("<%f, %d>\n", exh[i].first, exh[i].second);
    }

    printf("\n--------------------------------\n");

    std::vector<std::pair<double, int>> nearest1 = lsh.approximate_k_nearest_neighbors(images[1], 10); // Get the k nearest vectors to the first image
    for(int i = 0; i < (int)nearest1.size(); i++){
        printf("<%f, %d>\n", nearest1[i].first, nearest1[i].second);
    }
    printf("\n--------------------------------\n");

    std::vector<std::pair<double, int>> exh1 = exhaustive_nearest_neighbor_search(images, images[1], 10); // Get the k nearest vectors to the first image
    for(int i = 0; i < (int)exh1.size(); i++){
        printf("<%f, %d>\n", exh1[i].first, exh1[i].second);
    }

    printf("\n--------------------------------\n");

    // std::vector<std::pair<double, int>> exh2 = exhaustive_range_search(images, images[0], 2000); // Get the k nearest vectors to the first image
    // for(int i = 0; i < (int)exh2.size(); i++){
    //     printf("<%f, %d>\n", exh2[i].first, exh2[i].second);
    // }

    return 0;
}