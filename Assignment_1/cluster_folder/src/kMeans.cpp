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
#include "lsh.h"
#include "hypercube.h"

class Cluster{
    std::shared_ptr<ImageVector> Centroid;
    std::vector<std::shared_ptr<ImageVector>> Points;

    public:
    void set_centroid(std::shared_ptr<ImageVector> centroid){
        this->Centroid = centroid;
    }

    void add_point(std::shared_ptr<ImageVector> point){
        (this->Points).push_back(point);
    }

    std::shared_ptr<ImageVector> get_centroid(){
        return this->Centroid;
    }

    std::shared_ptr<ImageVector> recalculate_centroid(){
        int j;
        int clusterSize = (int)(this->Points).size();
        if(clusterSize == 0){
            return std::make_shared<ImageVector>(-1, std::vector<double>(0)); // Return a virtual point
        }
        std::vector<double> sum = (this->Points[0])->get_coordinates(); // Get the first point as initialization
        for(int i = 1; i < clusterSize; i++){ // Sum the vectors
            for(j = 0; j < (int)sum.size(); j++){ // Add each coordinate of the vectors
                sum[j] += (this->Points[i])->get_coordinates()[j];
            }
        }
        return std::make_shared<ImageVector>(-1, sum); // The fact that it is a virtual point is denoted by the -1 value of the number
    }

    void set_calulated_centroid(){
        this->Centroid = recalculate_centroid();
    }
};

class kMeans{
    int K; // Number of clusters
    std::vector<std::shared_ptr<ImageVector>> Points;
    std::vector<std::shared_ptr<ImageVector>> Centroids;
    std::vector<std::vector<std::shared_ptr<ImageVector>>> Clusters;
    Random R;

    public:
    kMeans(int k, std::vector<std::shared_ptr<ImageVector>> points){ // Needs the number of clusters and the dataset
        this->K = k;
        this->Points = points;

        // ------------------------------------------------------------------------------ //
        // ------------------------------ Initialization++ ------------------------------ //
        // ------------------------------------------------------------------------------ //

        printf("Initialization++\n");
        fflush(stdout);

        // -------------------------- Defines ------------------------ //
        int i, j, pointsNumber;
        double minDistance, minDistanceSquared, randomDistance, maxDistance;
        double sumOfSquaredDistances;
        std::priority_queue<double, std::vector<double>, std::greater<double>> distancesFromCentroids;
        std::priority_queue<double, std::vector<double>, std::less<double>> allDistances;
        std::vector<std::pair<double, int>> minDistances; // Save the max range that the point covers, and its index
        // -------------------------- Defines ------------------------ //

        // Find a good approximate of the max distance between two points for normalization

        printf("Approximating max distance\n");
        fflush(stdout);

        for(i = 0; i < (int)Points.size(); i++){
            allDistances.push(eucledian_distance((this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates(), (this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates()));
        }
        maxDistance = allDistances.top(); // Get the max distance

        printf("Max distance approximation: %f\n", maxDistance);
        fflush(stdout);

        // Assign the first centroid at random
        int firstCentroidIndex = R.generate_int_uniform(0, (int)(this->Points).size() - 1); // Get a random index for the first centroid
        std::shared_ptr<ImageVector> firstCentroid = this->Points[firstCentroidIndex]; // Get the first centroid
        this->Centroids.push_back(firstCentroid); // Add the first centroid to the centroids vector

        printf("Assigning Centroids... ");
        fflush(stdout);

        // Assign the rest of the centroids
        while((int)Centroids.size() < K){   
            // Resets
            sumOfSquaredDistances = 0; // Reset the sum of squared distances
            minDistances.clear(); // Reset the vector of minimum distances
            distancesFromCentroids = std::priority_queue<double, std::vector<double>, std::greater<double>>(); // Reset the priority queue of distances from centroids

            // Creating the biased probabilities
            for(i = 0; i < (int)(this->Points).size(); i++){ // For each point, calculate the distance from the nearest centroid
                for(j = 0; j < (int)(this->Centroids).size(); j++){
                    if((this->Points)[i] != (this->Centroids)[j]){ // If the point is a centroid, ignore it
                        distancesFromCentroids.push(eucledian_distance((this->Points)[i]->get_coordinates(), (this->Centroids)[j]->get_coordinates()));
                    }
                }
                minDistance = distancesFromCentroids.top(); // Pop the first element, which least distance from a centroid
                minDistance /= maxDistance; // Normalize the distance
                minDistanceSquared = minDistance * minDistance; // Get the square of the distance
                sumOfSquaredDistances += minDistanceSquared ; // Add the square of the distance to the sum of squared distances

                minDistances.push_back(std::make_pair(sumOfSquaredDistances, (this->Points)[i]->get_number())); // Add the distance to the vector of minimum distances
            }

            // Finding the cetroid
            randomDistance = R.generate_double_uniform(0, sumOfSquaredDistances); // Generate a random number between 0 and the sum of squared distances
            if(randomDistance >= 0 && randomDistance < minDistances[1].first){ // If the random number is 0, then the point is the centroid
                pointsNumber = minDistances[0].second; // Get the number of the point
                this->Centroids.push_back(this->Points[pointsNumber]); // Add the point to the centroids vector MAYBE ADD A FUNCTION THAT RETURNS A POINT GIVEN A NUMBER AND PUSH BACK THAT?
                continue;
            }
            for(i = 1; i < (int)(minDistances).size() - 1; i++){ // Find the point that corresponds to the random number
                if(randomDistance <= minDistances[i + 1].first && randomDistance > minDistances[i].first ){
                    pointsNumber = minDistances[i + 1].second;
                    this->Centroids.push_back(this->Points[pointsNumber]); // Add the point to the centroids vector
                    break;
                }
            }
        }
        printf("Done \n");
        fflush(stdout);
    }

    std::vector<std::shared_ptr<ImageVector>> get_centroids(){
        return this->Centroids;
    }
};


template <typename T>
std::vector<std::pair<double, int>> call(std::vector<std::pair<double, int>>(T::*func)(std::shared_ptr<ImageVector>, double), T& obj, std::shared_ptr<ImageVector> image, double r) {
    return (obj.*func)(image, r);
}

int main(void){
    std::vector<std::shared_ptr<ImageVector>> dataset = read_mnist_images("/home/xv6/Desktop/Project2023/Assignment_1/in/query.dat", 0);
    kMeans kmeans(10, dataset);
    printf("Number of centroids: %d\n", (int)kmeans.get_centroids().size());
    for(int i = 0; i < (int)kmeans.get_centroids().size(); i++){
        printf("Centroid %d: %d\n", i, kmeans.get_centroids()[i]->get_number());
    }

    LSH lsh(4,5,MODULO,LSH_TABLE_SIZE);
    lsh.load_data(dataset);

    std::vector<std::pair<double, int>> nearest_approx = call(&LSH::approximate_range_search, lsh, dataset[0], 2000.0);

    printf("Number of points: %d\n", (int)nearest_approx.size());

    return 0;
}
