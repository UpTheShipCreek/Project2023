#include "lsh.h"

hFunction::hFunction(){
    this->V = Rand.generate_vector_normal(DIMENSIONS, MEAN, STANDARD_DEVIATION); // The N(0,1) distribution
    this->T = Rand.generate_double_uniform(0.0, this->W); 
    // So we were explicitly instructed to use the uniform(0,W) distribution for t and N(0,1) for the values of v, 
    // but to also ensure that (p*v + t) is not negative?
    // Why are we allowing negative values in the first place then? 
    // Why are we not shifting the normal distribution to N(1,1) and the uniform distribution to (1, W+1) 
    // so as to not have to worry about negative values?
}

double hFunction::evaluate_point(std::vector<double> p){ // h(p) = (p*v + t)/w
    double product = std::inner_product(p.begin(), p.end(), (this->V).begin(), 0); 
    
    double result = (product + this->T)/ this->W;

    return (int)std::floor(result); // Casting the result into into so that we may operate it with other ints
}




gFunction::gFunction(int k, int m){
    this->K = k;
    this->M = m;
    for(int i = 0; i < this->K; i++){
        std::shared_ptr<hFunction> h = std::make_shared<hFunction>(); // Create a new h function, I used a pointer might not need it, I just said to myself after OOP that I'd always use pointers instead of storing objects directly, something that I haven't followed here generally
        (this->H).push_back(h); // Save its pointer to the vector
        (this->R).push_back(Rand.generate_int_uniform(1, 100)); // Generate and save the r value
    }
}

int gFunction::evaluate_point(std::vector<double> p){
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



HashTable::HashTable(int num, int k, int m){ // Constructor
    this->NumberOfBuckets = num;
    this->HashFunction = std::make_shared<gFunction>(k, m);
}

bool HashTable::same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2){ // Compares the id of two images, used in the querying trick
    return NumberToId[image1->get_number()] == NumberToId[image2->get_number()];
}

void HashTable::insert(std::shared_ptr<ImageVector> image){ // Insert an image to the hash table and save its id
    std::vector<double> p = image->get_coordinates();
    int id = HashFunction->evaluate_point(p);

    NumberToId[image->get_number()] = id;

    int bucketId = id % NumberOfBuckets;

    if (Table.find(bucketId) == Table.end()) {
        Table[bucketId] = std::vector<std::shared_ptr<ImageVector>>();
    }

    Table[bucketId].push_back(image);
}

const std::vector<std::shared_ptr<ImageVector>>& HashTable::get_bucket_of_image(std::shared_ptr<ImageVector> image){ // Returns the bucket a specific image resides in 
    int bucketId = NumberToId[image->get_number()] % NumberOfBuckets;
    return Table[bucketId];
}




  
LSH::LSH(int l, int k, int modulo, int tableSize){
    printf("Creating LSH... ");
    fflush(stdout);
    this->L = l;
    this->K = k;
    this->M = modulo;

    Tables.reserve(L); // Supposed to be making it faster, couldn't tell the difference (guessing the bottleneck is elsewhere)
    for (int i = 0; i < L; i++) {
        Tables.push_back(std::make_shared<HashTable>(tableSize, K, M));
    }
    printf("Done \n");
    fflush(stdout);
}

void LSH::load_data(std::vector<std::shared_ptr<ImageVector>> images){ // Load the data to the LSH
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

std::vector<std::pair<double, int>> LSH::approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest){
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

std::vector<std::pair<double, int>> LSH::approximate_range_search(std::shared_ptr<ImageVector> image, double r){
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