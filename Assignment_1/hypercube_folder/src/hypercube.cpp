#include "hypercube.h"

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
    std::vector<int> result;
    
    for (int i = 0; i < dimensions; i++) {
        int flippedBit = input ^ (1 << i);
        result.push_back(flippedBit);
    }
    
    return result;
}

std::vector<int> get_probes(int number, int numberOfProbes, int dimensions){
    std::set<int> uniqueProbes;  // So that we can ignore duplicates
    std::vector<int> probes;

    if (numberOfProbes <= 0){
        probes.push_back(number);
        return probes;
    } 
    else{
        std::vector<int> hammingDistanceOne = find_all_with_hamming_distance_one(number, dimensions);
        std::vector<int> temp;
        
        for (int i = 0; i < (int)hammingDistanceOne.size(); i++) {
            temp = get_probes(hammingDistanceOne[i], numberOfProbes - 1, dimensions); // Recursive call cause it is easier to think of it that way
            uniqueProbes.insert(temp.begin(), temp.end());  // Insert into the set to ensure uniqueness
        }
        probes.insert(probes.end(), uniqueProbes.begin(), uniqueProbes.end()); // Fill up the returning vector 
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
    //printf("HashCode: %d\n", hashCode);
    return hashCode;
}


HyperCube::HyperCube(int dimensions, int probes, int numberOfElementsToCheck){
    this->M = numberOfElementsToCheck;
    this->K = dimensions;
    this->Probes = probes;

    std::shared_ptr<HashFunction> hashFunction = std::make_shared<HypercubeHashFunction>(this->K);
    int numberOfBuckets = 1 << K;

    //printf("Number of buckets: %d\n", numberOfBuckets);

    this->Table = std::make_shared<HashTable>(numberOfBuckets, hashFunction);
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

    // Get all the (this->Probes)# of probes of a (this->K)-dimensional hypercube
    probes = get_probes(bucketId, this->Probes, this->K);
    //printf("Probes of %d are : \n", bucketId);
    for(i = 0; i < (int)(probes.size()); i++){
        //printf("%d\n", probes[i]);
    }

    // For each probe 
    for(i = 0; i < (int)(probes.size()); i++){
        // Get the bucket
        bucket = (this->Table)->get_bucket_from_bucket_id(probes[i]);  

        // Search the bucket for the nearest neighbors
        j = 0;
        //while(j < (int)(bucket.size()) && visitedPointsCounter < (this->M)){
        while(j < (int)(bucket.size())){
            //printf("Bucket size is: %ld\n", bucket.size());

            visitedPointsCounter++;
            prospectImageNumber = bucket[j]->get_number();

            // Ignore comparing with itself
            if(prospectImageNumber != queryImageNumber){ 
                // dist(p,q)
                distance = eucledian_distance(image->get_coordinates(), bucket[j]->get_coordinates());

                //printf("Found distance: %f\n", distance);

                nearest.push(std::make_pair(distance, prospectImageNumber));

                // Keep the number of nearest neighbors the correct size
                if((int)nearest.size() > numberOfNearest){
                    nearest.pop();
                }
            }
            j++;
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

