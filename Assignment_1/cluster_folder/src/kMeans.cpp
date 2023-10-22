#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>
#include <cfloat>
#include <functional>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"
#include "lsh.h"
#include "hypercube.h"

#define NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE 0.8 // If at 80% of the clusters are converged then we have converged
#define DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE 0.01 // If the change in the distance is less than 1% of the max distance between two points in our dataset then we have converged
#define CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE 0.2 // Taking into account the percentage of the change of the change of distance between two epochs

double round_up_to_nearest_order_of_magnitude(double number){
    double order = std::pow(10, std::floor(std::log10(number)));
    return ceil(number / order) * order;
}

class Cluster{
    std::shared_ptr<ImageVector> Centroid;
    std::vector<std::shared_ptr<ImageVector>> Points;

    public:
    Cluster();

    Cluster(std::shared_ptr<ImageVector> centroid){
        this->Centroid = centroid;
    }

    Cluster(std::vector<std::shared_ptr<ImageVector>> points){
        this->Points = points;
    }

    Cluster(std::shared_ptr<ImageVector> centroid, std::vector<std::shared_ptr<ImageVector>> points){
        this->Centroid = centroid;
        this->Points = points;
    }

    void set_centroid(std::shared_ptr<ImageVector> centroid){
        this->Centroid = centroid;
    }

    void add_point(std::shared_ptr<ImageVector> point){
        (this->Points).push_back(point);
    }

    std::shared_ptr<ImageVector> get_centroid(){
        return this->Centroid;
    }

    std::vector<std::shared_ptr<ImageVector>> get_points(){
        return this->Points;
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
        for(j = 0; j < (int)sum.size(); j++){ // Divide each coordinate by the number of points
            sum[j] /= clusterSize;
        }
        return std::make_shared<ImageVector>(-1, sum); // The fact that it is a virtual point is denoted by the -1 value of the number
    }

    void calculate_and_set_centroid(){
        this->Centroid = recalculate_centroid();
    }
};

class kMeans{
    int K; // Number of clusters
    std::vector<std::shared_ptr<ImageVector>> Points;
    std::vector<std::shared_ptr<Cluster>> Clusters;
    Random R;
    using AssignmentFunction = std::function<void()>; // Define a function pointer type for the mac_queen method
    double MaxDist;

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

        std::shared_ptr<ImageVector> centroid;
        std::shared_ptr<Cluster> cluster;

        std::priority_queue<double, std::vector<double>, std::greater<double>> distancesFromCentroids;
        std::priority_queue<double, std::vector<double>, std::less<double>> allDistances;
        std::vector<std::pair<double, int>> minDistances; // Save the max range that the point covers, and its index
        // -------------------------- Defines ------------------------ //

        // Find a good approximate of the max distance between two points for normalization

        printf("Approximating max distance\n");
        fflush(stdout);

        for(i = 0; i < (int)std::sqrt(Points.size()); i++){ // Get the distances of some points we don't need many since we are rounding up anyhow
            allDistances.push(eucledian_distance((this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates(), (this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates()));
        }

        

        maxDistance = round_up_to_nearest_order_of_magnitude(allDistances.top()); // Get the max distance
        this->MaxDist = maxDistance;

        printf("Max distance approximation: %f\n", maxDistance);
        fflush(stdout);

        // Assign the first centroid at random
        int firstCentroidIndex = R.generate_int_uniform(0, (int)(this->Points).size() - 1); // Get a random index for the first centroid
        std::shared_ptr<ImageVector> firstCentroid = this->Points[firstCentroidIndex]; // Get the first centroid

        cluster = std::make_shared<Cluster>(firstCentroid); // Make a cluster with the first centroid
        this->Clusters.push_back(cluster); // Save it in our clusters vector

        printf("Assigning Centroids... ");
        fflush(stdout);

        // Assign the rest of the centroids
        while((int)(this->Clusters).size() < K){   
            // Resets
            sumOfSquaredDistances = 0; // Reset the sum of squared distances
            minDistances.clear(); // Reset the vector of minimum distances
            distancesFromCentroids = std::priority_queue<double, std::vector<double>, std::greater<double>>(); // Reset the priority queue of distances from centroids

            // Creating the biased probabilities
            for(i = 0; i < (int)(this->Points).size(); i++){ // For each point, calculate the distance from the nearest centroid
                for(j = 0; j < (int)(this->Clusters).size(); j++){

                    centroid = (this->Clusters)[j]->get_centroid(); // Get the centroid of the cluster

                    if((this->Points)[i] != centroid){ // If the point is a centroid, ignore it
                        distancesFromCentroids.push(eucledian_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates()));
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
                cluster = std::make_shared<Cluster>(this->Points[pointsNumber]); // Make a cluster with the point as a centroid MAYBE ADD A FUNCTION THAT RETURNS A POINT GIVEN A NUMBER AND PUSH BACK THAT?
                this->Clusters.push_back(cluster); // Save it in our clusters vector
                continue;
            }
            for(i = 1; i < (int)(minDistances).size() - 1; i++){ // Find the point that corresponds to the random number
                if(randomDistance <= minDistances[i + 1].first && randomDistance > minDistances[i].first ){
                    pointsNumber = minDistances[i + 1].second;
                    cluster = std::make_shared<Cluster>(this->Points[pointsNumber]); // Make a cluster with the point as a centroid
                    this->Clusters.push_back(cluster); // Save it in our clusters vector
                    break;
                }
            }
        }
        printf("Done \n");
        fflush(stdout);
    }

    std::shared_ptr<Cluster> get_nearest_cluster(std::shared_ptr<ImageVector> point){
        int j;
        double distance;
        double minDinstace = DBL_MAX;
        std::shared_ptr<ImageVector> tempCentroid;
        std::shared_ptr<Cluster> nearestCluster;
    
        for(j = 0; j < (int)(this->Clusters).size(); j++){
            tempCentroid = (this->Clusters)[j]->get_centroid();
            distance = eucledian_distance(point->get_coordinates(), tempCentroid->get_coordinates()); // Calculate the distance from each centroid
            if(distance < minDinstace){
                minDinstace = distance; // Get the minimum distance
                nearestCluster = (this->Clusters)[j]; // Get the nearest cluster
            }
        }
        return nearestCluster;
    }

    std::vector<std::shared_ptr<ImageVector>> get_centroids(){
        std::vector<std::shared_ptr<ImageVector>> centroids;
        for(int i = 0; i < (int)(this->Clusters).size(); i++){
            centroids.push_back((this->Clusters)[i]->get_centroid());
        }
        return centroids;
    }

    void lloyds_assigment(){ // Lloyds-type assignment
        int i, j;
        std::shared_ptr<ImageVector> nearestCentroid;
        std::shared_ptr<Cluster> nearestCluster;
        for(i = 0; i < (int)(this->Points).size(); i++){
            nearestCluster = get_nearest_cluster((this->Points)[i]); // Get the nearest centroid to the point
            nearestCentroid = nearestCluster->get_centroid(); // Get the centroid
            for(j = 0; j < (int)(this->Clusters).size(); j++){
                if((this->Clusters)[j]->get_centroid() == nearestCentroid){ // If the centroid is the nearest centroid
                    (this->Clusters)[j]->add_point((this->Points)[i]); // Add the point to the cluster
                    break;
                }
            }
        }
    }

    void reverse_assignment(std::shared_ptr<ApproximateMethods> method){
        int i, j;
        double dinstaceBetweenCentroids, radius;
        double minDistanceBetweenCentroids = DBL_MAX;
        double maxDistanceBetweenCentroids = DBL_MIN;
        std::shared_ptr<ImageVector> centroid;
        std::shared_ptr<Cluster> nearestCluster;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> inRangeImages;

        
        // Create a new structure that keeps track of the assigned images
        std::shared_ptr<ImageVector> tempImageVector;
        std::unordered_set<std::shared_ptr<ImageVector>> unassignedImages;

        for(i = 0; i < (int)(this->Points).size(); i++){
            tempImageVector = std::make_shared<ImageVector>((this->Points)[i]->get_number(), (this->Points)[i]->get_coordinates());
            unassignedImages.insert(tempImageVector);
        }
        
        // Find the minimum and the maximum distance between centroids
        for(i = 0; i < (int)Clusters.size(); i++){
            for(j = 0; j < (int)Clusters.size(); j++){
                if(i != j){
                    dinstaceBetweenCentroids = eucledian_distance((this->Clusters)[i]->get_centroid()->get_coordinates(), (this->Clusters)[j]->get_centroid()->get_coordinates());
                    if(dinstaceBetweenCentroids < minDistanceBetweenCentroids){
                        minDistanceBetweenCentroids = dinstaceBetweenCentroids;
                    }
                    if(dinstaceBetweenCentroids > maxDistanceBetweenCentroids){
                        maxDistanceBetweenCentroids = dinstaceBetweenCentroids;
                    }
                }
            }
        }

        
        // Start the assignation 
        radius = minDistanceBetweenCentroids / 2;
        while(!unassignedImages.empty() && radius < maxDistanceBetweenCentroids){ // Keep going until you have assigned all the points or the radius is bigger than the max distance between centroids
            // Assign imagevectors to clusters
            for(i = 0; i < (int)Clusters.size(); i++){
                centroid = (this->Clusters)[i]->get_centroid();
                inRangeImages = method->approximate_range_search_return_images(centroid, radius);
                for(j = 0; j < (int)inRangeImages.size(); j++){
                    if(unassignedImages.find((inRangeImages[j].second)) != unassignedImages.end()){ // If the point is unassingned
                        (this->Clusters)[i]->add_point(inRangeImages[j].second); // Add the point to the cluster
                        unassignedImages.erase((inRangeImages[j].second));  // Delete the image from the unassigned images
                    }
                }
            }
            radius *= 2; // Increase the radius
        }

        // Assign the remaining points to the nearest centroid
        for (const auto& item : unassignedImages) {
            nearestCluster = get_nearest_cluster(item);
            nearestCluster->add_point(item);
        }
    }

    void mac_queen(AssignmentFunction assignment){
        int i, clustersConvergedCounter;
        int clusterNumberConvergenceTolerance = int((double)(this->Clusters).size() * NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE); // If at 80% of the clusters are converged then we have converged
        bool hasConverged = false;

        std::vector<double> oldDistanceDifferenceVector((this->Clusters).size(), 0.0); // I need to initialize this with the distance of each centroid from the (0, 0...0) point
        double newDistanceDifference;
        double distanceDifferenceAsMaxPercentageTolerance = this->MaxDist * DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE; //  If the change in the distance is less than 1% of the max distance between two points in our dataset then we have converged
        
        double changeOfDistanceDifferencePercentage;

        std::vector<double> zeroVector(((this->Clusters)[0]->get_centroid()->get_coordinates()).size(), 0.0);

        printf("Need at least %d clusters to converge\n", clusterNumberConvergenceTolerance);
        printf("Tolerance percentage: %f\n", distanceDifferenceAsMaxPercentageTolerance);

        std::shared_ptr<ImageVector> newCentroid;

        for(i = 0; i < (int)(this->Clusters).size(); i++){ // Initialize the oldDistanceDifference
            oldDistanceDifferenceVector[i] = eucledian_distance((this->Clusters)[i]->get_centroid()->get_coordinates(),zeroVector);
        }

        do{
            assignment(); // Assign the points to the clusters
            clustersConvergedCounter = 0;
            for(i = 0; i < (int)(this->Clusters).size(); i++){ // Recalculate the centroids
                newCentroid = (this->Clusters)[i]->recalculate_centroid();
                newDistanceDifference = eucledian_distance(newCentroid->get_coordinates(), (this->Clusters)[i]->get_centroid()->get_coordinates());

                changeOfDistanceDifferencePercentage = (oldDistanceDifferenceVector[i] - newDistanceDifference) / oldDistanceDifferenceVector[i];
                
                oldDistanceDifferenceVector[i] = newDistanceDifference;

                printf("%d distance: %f and percentage change: %f\n", i, newDistanceDifference, changeOfDistanceDifferencePercentage);

                if(newDistanceDifference < distanceDifferenceAsMaxPercentageTolerance ||  changeOfDistanceDifferencePercentage < CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE){ // If the distance is bigger than the tolerance
                    clustersConvergedCounter++;
                }

                (this->Clusters)[i]->set_centroid(newCentroid);
            }
            printf("%d clusters converged\n", clustersConvergedCounter);
            if(clustersConvergedCounter >= clusterNumberConvergenceTolerance){ // Check if our 80% floor has been reached
                hasConverged = true;
            }
        }while(!hasConverged);
    }
};

std::vector<std::pair<double, int>> approximate_range_search(ApproximateMethods* method, std::shared_ptr<ImageVector> image, double r) {
    return method->approximate_range_search(image, r);
}

int main(void){
    
    std::vector<std::shared_ptr<ImageVector>> dataset = read_mnist_images("/home/xv6/Desktop/Project2023/Assignment_1/in/query.dat", 0);

    std::shared_ptr<kMeans> kmeans;
    kmeans = std::make_shared<kMeans>(10, dataset);

    double sumOfDistances = 0;
    printf("Number of centroids: %d\n", (int)kmeans->get_centroids().size());
    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){
        printf("Centroid %d: %d\n", i, kmeans->get_centroids()[i]->get_number());
    }

    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){ 
        for(int j = 0; j < (int)kmeans->get_centroids().size(); j++){ 
            sumOfDistances += eucledian_distance(kmeans->get_centroids()[i]->get_coordinates(), kmeans->get_centroids()[j]->get_coordinates());
        }
    }
    printf("Metric: %f\n", sumOfDistances);

    std::shared_ptr<LSH> lsh;
    lsh = std::make_shared<LSH>(4,5,MODULO,LSH_TABLE_SIZE);
    lsh->load_data(dataset);


    // kmeansInstance.mac_queen([&kmeansInstance](){ kmeansInstance.lloyds_assignment(); });
    kmeans->mac_queen([kmeans](){kmeans->lloyds_assigment();});
    
    // kmeansInstance.mac_queen(std::bind(&kMeans::reverse_assignment, &kmeansInstance, method));
    kmeans->mac_queen(std::bind(&kMeans::reverse_assignment, kmeans, lsh));

    return 0;
}
