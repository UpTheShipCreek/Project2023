#include "hypercube.h"

long double factorial(int n){
    if(n < 0) return 0;

    int i;
    long double  result = 1; 

    for(i = 1; i <= n; i++){
        result *= i;
    }
    return result;
}

int calculate_number_of_probes_given_maximum_hamming_distance(int maximumHammingDistance, int dimensions){
    int probes = 0;
    for(int i = 1; i <= maximumHammingDistance; i++){
        if(i > dimensions) break;
        probes += factorial(dimensions)/(factorial(i)*factorial(dimensions-i));
    }  
    return probes;
}

int hamming_distance(int x, int y){ 
    int dist = 0;

    // The ^ operators sets to 1 only the bits that are different
    for (int val = x ^ y; val > 0; dist++)
    {
        // We then count the bit set to 1 using the Peter Wegner way
        val = val & (val - 1); // Set to zero val's lowest-order 1
    }

    // Return the number of differing bits
    return dist;
}

std::vector<int> find_all_with_hamming_distance_one(int input, int dimensions){
    int neighbor;
    std::vector<int> result;
    
    for (int i = 0; i < dimensions; i++){
        neighbor = input ^ (1 << i); // The XOR operation with a 1 at the i-th position flips the i-th bit, so we end up finding all the direct neighbors i.e. numbers with one flipped bit
        result.push_back(neighbor);
    }
    return result;
}

std::vector<int> get_probes(int number, int maxHammingDistance, int dimensions){ // Changed it to BFs cause with the probe changes I was getting bad results
    int current, levelSize;

    std::vector<int> hammingDistanceOne;
    std::vector<int> probes;
    std::unordered_set<int> uniqueProbes; // So that we can ignore duplicates
    std::queue<int> bfsQueue;

    probes.push_back(number);
    bfsQueue.push(number);

    while(maxHammingDistance > 0){
        levelSize = bfsQueue.size();
        while (levelSize > 0){
            current = bfsQueue.front();
            bfsQueue.pop();

            hammingDistanceOne = find_all_with_hamming_distance_one(current, dimensions);

            for(int i = 0; i < (int)hammingDistanceOne.size(); i++){
                if(uniqueProbes.insert(hammingDistanceOne[i]).second){
                    bfsQueue.push(hammingDistanceOne[i]);
                    probes.push_back(hammingDistanceOne[i]);
                }
            }
            levelSize--;
        }
        maxHammingDistance--;
    }
    return probes;
}
HypercubeHashFunction::HypercubeHashFunction(int k){ // Constructor 
    std::shared_ptr<hFunction> h;
    std::shared_ptr<fFunction> f;
    this->K = k;
    for(int i = 0; i < this->K; i++){
        h = std::make_shared<hFunction>();
        H.push_back(h);

        f = std::make_shared<fFunction>();
        F.push_back(f);
    }
}
int HypercubeHashFunction::evaluate_point(std::vector<double> p){
    int bDigit;
    int hashCode = 0;
    for(int i = 0; i < this->K; i++){
        bDigit = F[i]->evaluate_point(H[i]->evaluate_point(p));
        hashCode <<= 1; // shift so that we have some space for the next digit
        hashCode |= bDigit; // save the code of the particular projection
    }
    return hashCode;
}

HyperCube::HyperCube(int dimensions, int probes, int numberOfElementsToCheck, Metric* metric){
    this->M = numberOfElementsToCheck;
    this->K = dimensions;
    this->Probes = probes;
    this->Hmetric = metric;
    this->MaxHammingDistance = 0;

    std::shared_ptr<HashFunction> hashFunction = std::make_shared<HypercubeHashFunction>(this->K);
    int numberOfBuckets = 1 << K; // Essentially 2^K

    this->Table = std::make_shared<HashTable>(numberOfBuckets, hashFunction);

    // Calculate the maximum hamming distance to look at given that we want to visit at most this->Probes vertices
    // by finding the smallest maxHammingDistance that represents more vertices than this->Probes
    int numOfProbes = 0;
    // Run until you hit the max number of possible probes given the dimensions or you hit the max number of probes that the user wants
    do{
        numOfProbes = calculate_number_of_probes_given_maximum_hamming_distance(this->MaxHammingDistance, this->K);
        this->MaxHammingDistance++;
    }while(numOfProbes < this->Probes && numOfProbes < (1 << this->K) -1 ); // It's -1 cause we don't want to count the vertex itself
}
void HyperCube::load_data(std::vector<std::shared_ptr<ImageVector>> images){
    printf("Loading data into the hypercube... ");
    fflush(stdout);
    for (int i = 0; i < (int)(images.size()); i++){
        (this->Table)->insert(images[i]);
    }
    printf("Done\n");
    fflush(stdout);
}
std::vector<std::pair<double, int>> HyperCube::approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest){
    int i, j, prospectImageNumber;
    double distance;
    int visitedPointsCounter = 0;
    int queryImageNumber = image->get_number();

    // This will be saving all the bucket_ids/hypercube vertices that we will be visiting
    std::vector<int> probes;
    std::vector<std::shared_ptr<ImageVector>> bucket;

    // I will be using a priority queue to keep the k nearest neighbors
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> nearest;

    // Let b ← Null; db ← ∞; initialize k best candidates and distances;
    std::vector<std::pair<double, int>> nearestImages;
    
    // Get the bucket id
    int bucketId = Table->get_bucket_id_from_image_vector(image);

    probes = get_probes(bucketId, this->MaxHammingDistance, this->K);

    // For each probe / i.e. for each neighboring vertex of the hypercube within #probe steps
    // i = 0;
    // while(i < (int)(probes.size()) && visitedPointsCounter < (this->M)){
    for(i = 0; i < this->Probes; i++){
        // Get the bucket
        bucket = (this->Table)->get_bucket_from_bucket_id(probes[i]);  

        // Search the bucket for the nearest neighbors
        j = 0;
        while(j < (int)(bucket.size()) && visitedPointsCounter < (this->M)){
            visitedPointsCounter++;

            prospectImageNumber = bucket[j]->get_number();

            // Ignore comparing with itself
            if(prospectImageNumber != queryImageNumber){ 
                // dist(p,q)
                distance = Hmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());

                nearest.push(std::make_pair(distance, prospectImageNumber));

                // Keep the number of nearest neighbors the correct size
                if((int)nearest.size() > numberOfNearest){
                    nearest.pop();
                }
            }
            j++;
        }
        // i++;
    }
    // Fill up the a structure that we can return
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    std::vector<std::pair<double,int>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;
} 

std::vector<std::pair<double, int>> HyperCube::approximate_range_search(std::shared_ptr<ImageVector> image, double r){
    int i, j, prospectImageNumber;
    double distance;
    int visitedPointsCounter = 0;
    int queryImageNumber = image->get_number();

    // This will be saving all the bucket_ids/hypercube vertices that we will be visiting
    std::vector<int> probes;
    std::vector<std::shared_ptr<ImageVector>> bucket;

    // I will be using a priority queue to keep the k nearest neighbors
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> nearest;

    // Let b ← Null; db ← ∞; initialize k best candidates and distances;
    std::vector<std::pair<double, int>> nearestImages;

    // Get the bucket id
    int bucketId = Table->get_bucket_id_from_image_vector(image);

    // Get all the (this->Probes)# of probes of a (this->K)-dimensional hypercube
    probes = get_probes(bucketId, this->MaxHammingDistance, this->K);

    // For each probe / i.e. for each neighboring vertex of the hypercube within #probe steps
   i = 0; 
   while(i < this->Probes && i < (int)(probes.size())){
        // Get the bucket
        bucket = (this->Table)->get_bucket_from_bucket_id(probes[i]);  

        // Search the bucket for the nearest neighbors
        //for(j = 0; j < (int)(bucket.size()); j++){
        j = 0;
        while(j < (int)(bucket.size()) && visitedPointsCounter < (this->M)){
            visitedPointsCounter++;

            prospectImageNumber = bucket[j]->get_number();

            // Ignore comparing with itself
            if(prospectImageNumber != queryImageNumber){
                // dist(p,q)
                distance = Hmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());

                if(distance <= r){
                    nearest.push(std::make_pair(distance, prospectImageNumber));
                }
            }
            j++;
        }
        i++;
    }
    // Fill up the a structure that we can return
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    std::vector<std::pair<double,int>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> HyperCube::approximate_range_search_return_images(std::shared_ptr<ImageVector> image, double r){
    int i, j;
    double distance;
    int visitedPointsCounter = 0;

    std::pair<int, int> imageBucketIdAndId;

    // This will be saving all the bucket_ids/hypercube vertices that we will be visiting
    std::vector<int> probes;

    // The returned vector
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> inRangeImages;

    // The pointer of each bucket
    std::vector<std::shared_ptr<ImageVector>> bucket;

    // Get the bucket id and the image id
    imageBucketIdAndId = Table->virtual_insert(image);
    // printf("Virtual Insert: BucketId: %d, ImageId: %d\n", imageBucketIdAndId.first, imageBucketIdAndId.second);

    // Get all the (this->Probes)# of probes of a (this->K)-dimensional hypercube
    probes = get_probes(imageBucketIdAndId.first, this->Probes, this->K);

    // For each probe / i.e. for each neighboring vertex of the hypercube within #probe steps
    i = 0; 
    while(i < this->Probes && i < (int)(probes.size())){

        // printf("Probe: %d\n", probes[i]);

        // Get the bucket
        bucket = (this->Table)->get_bucket_from_bucket_id(probes[i]);  

        // Search the bucket for the nearest neighbors
        j = 0;
        while(j < (int)(bucket.size()) && visitedPointsCounter < (this->M)){
            visitedPointsCounter++;
            // Ignore comparing with itself
            if(image != bucket[j]){

                // dist(p,q)
                distance = Hmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());

                if(distance <= r){
                    inRangeImages.push_back(std::make_pair(distance, bucket[j]));
                }
            }
            j++;
        }
        i++;
    }
    return inRangeImages;
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> HyperCube::approximate_k_nearest_neighbors_return_images(std::shared_ptr<ImageVector> image, int numberOfNearest){
    int i, j;
    double distance;
    int visitedPointsCounter = 0;

    std::pair<int,int> imageBucketIdAndId;

    // This will be saving all the bucket_ids/hypercube vertices that we will be visiting
    std::vector<int> probes;

    // I will be using a priority queue to keep the k nearest neighbors
    std::priority_queue<std::pair<double, std::shared_ptr<ImageVector>>, std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, std::less<std::pair<double, std::shared_ptr<ImageVector>>>> nearest;

    // Let b ← Null; db ← ∞; initialize k best candidates and distances;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;

    // The pointer of each bucket
    std::vector<std::shared_ptr<ImageVector>> bucket;

    // Get the bucket id and the image id
    imageBucketIdAndId = Table->virtual_insert(image);

    // Get all the (this->Probes)# of probes of a (this->K)-dimensional hypercube
    probes = get_probes(imageBucketIdAndId.first, this->Probes, this->K);

     // For each probe / i.e. for each neighboring vertex of the hypercube within #probe steps
    i = 0; 
    while(i < this->Probes && i < (int)(probes.size())){
        // Get the bucket
        bucket = (this->Table)->get_bucket_from_bucket_id(probes[i]);  

        // Search the bucket for the nearest neighbors
        j = 0;
        while(j < (int)(bucket.size()) && visitedPointsCounter < (this->M)){
            visitedPointsCounter++;
            // Ignore comparing with itself
            if(image != bucket[j]){
                // dist(p,q)
                distance = Hmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());

                // if dist(q, p) < db = k-th best distance then b ← p; db ← dist(q, p), implemented with a priority queue
                nearest.push(std::make_pair(distance, bucket[j]));    

                if ((int)(nearest.size()) > numberOfNearest){
                    nearest.pop(); // Remove the largest image if we are out of space
                }
            }
            j++;
        }
        i++;
    }
    // Fill up the a structure that we can return
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;

}

