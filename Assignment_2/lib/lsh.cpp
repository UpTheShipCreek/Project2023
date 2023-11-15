#include "lsh.h"


LSH::LSH(int l, int k, double window, int tableSize, Metric* metric){
    this->L = l;
    this->K = k;
    this->W = window;
    this->Lmetric = metric;

    Tables.reserve(L);

    for (int i = 0; i < L; i++){
        std::shared_ptr<HashFunction> hashFunction = std::make_shared<gFunction>(this->K, this->W);
        Tables.push_back(std::make_shared<HashTable>(tableSize, hashFunction));
    }
}

LSH::LSH(int l, int k, int tableSize, Metric* metric){
        this->L = l;
        this->K = k;
        this->Lmetric = metric;

        Tables.reserve(L);

        for (int i = 0; i < L; i++){
            std::shared_ptr<HashFunction> hashFunction = std::make_shared<gFunction>(this->K, this->M);
            Tables.push_back(std::make_shared<HashTable>(tableSize, hashFunction));
        }
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
        bucket = Tables[i]->get_bucket_from_image_vector(image);
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
                distance = Lmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                
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
        bucket = Tables[i]->get_bucket_from_image_vector(image);
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
                distance = Lmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                if(distance < r){
                    inRangeImages.push_back(std::make_pair(distance, imageNumber)); // maybe add bucket[j] also
                }
            }
            // if large number of retrieved items (e.g. > 20L) then return
            if((int)inRangeImages.size() > 20*(this->L)) return inRangeImages;
        }
    }
    return inRangeImages;
}

// Turns out I did some assumptions making the LSH/Hypercube programs that weren't so good for the k-means. 
// Firstly I assummed that it was okay to treat every query as a query from within the dataset itself, so I just pushed the queries into the dataset as well
// and secondly I chose to not return the imagevector type at all but only the distance and the imagenumber
// This implemntation solves both of these problems; it assumes that the query is not from the dataset and it returns the imagevector type along with the distance
// Downside of this is that I had to implement some hashtable class methods that are arguably violating encapsulation 
std::vector<std::pair<double, std::shared_ptr<ImageVector>>> LSH::approximate_range_search_return_images(std::shared_ptr<ImageVector> image, double r){ 
    int i, j;
    double distance;

    std::pair<int,int> imageBucketIdAndId;

    // Ignore itself and every other image it has met before
    std::vector<std::shared_ptr<ImageVector>> ignore;
    std::vector<std::shared_ptr<ImageVector>>::iterator it; // Initializing the iteration variable
    ignore.push_back(image);

    // The returned vector
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> inRangeImages;

    // The pointer of each bucket
    std::vector<std::shared_ptr<ImageVector>> bucket;

    // for i from 1 to L do
    for(i = 0; i < L; i++){
        imageBucketIdAndId = Tables[i]->virtual_insert(image);
        bucket = Tables[i]->get_bucket_from_bucket_id(imageBucketIdAndId.first);
        // for each item p in bucket gi (q) do
        for(j = 0; j < (int)bucket.size(); j++){ // For each image in the bucket

            // See if we have encountered it before
            it = std::find(ignore.begin(), ignore.end(), bucket[j]);

            if(Tables[i]->get_image_id(bucket[j]) == imageBucketIdAndId.second && (it == ignore.end())){ // Query trick + ignore the images we have encountered before
                // Ignore the image if you find it again
                ignore.push_back(bucket[j]);

                // if dist(q, p) < r then output p
                distance = Lmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                if(distance <= r){
                    inRangeImages.push_back(std::make_pair(distance, bucket[j])); // maybe add bucket[j] also
                }
            }
        }
    }
    return inRangeImages;
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> LSH::approximate_k_nearest_neighbors_return_images(std::shared_ptr<ImageVector> image, int numberOfNearest){
    int i, j;
    double distance;

    std::pair<int,int> imageBucketIdAndId;

    // Ignore itself and every other image it has met before
    std::vector<std::shared_ptr<ImageVector>> ignore;
    std::vector<std::shared_ptr<ImageVector>>::iterator it; // Initializing the iteration variable
    ignore.push_back(image);

    // I will be using a priority queue to keep the k nearest neighbors
    std::priority_queue<std::pair<double, std::shared_ptr<ImageVector>>, std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, std::less<std::pair<double, std::shared_ptr<ImageVector>>>> nearest;

    // Let b ← Null; db ← ∞; initialize k best candidates and distances;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;

    // The pointer of each bucket
    std::vector<std::shared_ptr<ImageVector>> bucket;
    
    // for i from 1 to L do
    for(i = 0; i < this->L; i++){
        imageBucketIdAndId = Tables[i]->virtual_insert(image);

        bucket = Tables[i]->get_bucket_from_bucket_id(imageBucketIdAndId.first);
        // for each image in the bucket 
        for(j = 0; j < (int)(bucket.size()); j++){ 

            // See if we have encountered the image before before
            it = std::find(ignore.begin(), ignore.end(), bucket[j]);
            
            if(Tables[i]->get_image_id(bucket[j]) == imageBucketIdAndId.second && (it == ignore.end())){ // Query trick + ignore the images we have encountered before
                // Ignore the image if you find it again
                ignore.push_back(bucket[j]);

                // dist(p,q)
                distance = Lmetric->calculate_distance(image->get_coordinates(), bucket[j]->get_coordinates());
                
                // if dist(q, p) < db = k-th best distance then b ← p; db ← dist(q, p), implemented with a priority queue
                nearest.push(std::make_pair(distance, bucket[j]));    

                if ((int)(nearest.size()) > numberOfNearest){
                    nearest.pop(); // Remove the largest image if we are out of space
                }  
            }
        }
    }
    // Fill up the a structure that we can return
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;
}