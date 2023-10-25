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
#include <map>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"
#include "lsh.h"
#include "hypercube.h"

#define NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE 0.9 // If at 90% of the clusters are converged then we have converged
#define DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE 0.01 // If the change in the distance is less than 1% of the max distance between two points in our dataset then we have converged
#define CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE 0.95 // Taking into account the percentage of the change of the change of distance between two epochs

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

    // Centroid_[n+1] = (N/N+1) Centroid_[n] + newPoint/N+1
    void add_point_and_set_centroid(std::shared_ptr<ImageVector> point){
        (this->Points).push_back(point);
        double numberOfPoints = ((this->Points).size());
        double fraction = (numberOfPoints) / (numberOfPoints + 1);

        double temp;
        double newvalue;

        if(this->Centroid->get_number() != -1){ // If the centroid is not virtual we need to create another one in order not to change the coordinates of the actual dataset image
            std::shared_ptr<ImageVector> centroidCopy = std::make_shared<ImageVector>(-1, this->Centroid->get_coordinates());
            this->Centroid = centroidCopy;
        }
        for (int i = 0; i < (int)(this->Centroid)->get_coordinates().size(); i++){
            temp =  (this->Centroid)->get_coordinates()[i];
            newvalue = (fraction * temp) + (point->get_coordinates()[i] / (numberOfPoints + 1));
            
            (this->Centroid)->get_coordinates()[i] = newvalue;
        }
    }

    std::shared_ptr<ImageVector>& get_centroid(){
        return this->Centroid;
    }

    std::vector<std::shared_ptr<ImageVector>>& get_points(){
        return this->Points;
    }

    std::shared_ptr<ImageVector> recalculate_centroid(){
        int j;
        int clusterSize = (int)(this->Points).size();

        if(clusterSize == 0){
            return this->Centroid;
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
        if(this->Centroid->get_number() != -1){ // If the centroid is not virtual we need to create another one in order not to change the coordinates of the actual dataset image
            std::shared_ptr<ImageVector> centroidCopy = std::make_shared<ImageVector>(-1, sum);
            this->Centroid = centroidCopy;
        } 
        else{
            this->Centroid->get_coordinates() = sum;
        }
        return this->Centroid;
    }
};

class kMeans{
    int K; // Number of clusters
    std::vector<std::shared_ptr<ImageVector>> Points;
    std::vector<std::shared_ptr<Cluster>> Clusters;
    std::vector<double> CenterMass;

    Random R;
    using AssignmentFunction = std::function<void()>; // Define a function pointer type for the mac_queen method
    double MaxDist;
    Metric* Kmetric;


    public:
    kMeans(int k, std::vector<std::shared_ptr<ImageVector>> points, Metric* metric){ // Needs the number of clusters and the dataset
        this->K = k;
        this->Points = points;
        this->Kmetric = metric;
        // Using this to calculate the first centroid distance differences, i.e. the difference between the center of mass and the assigned from initialization++
        // It is part of this convergence condition CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE
        this->CenterMass.resize((this->Points)[0]->get_coordinates().size(), 0); // Initialize the center of mass vector with 0s

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

        std::vector<double> coordinates1, coordinates2;
        int numberOfPointsToCheck = (int)std::sqrt(Points.size()); // Get the distances of some points we don't need many since we are rounding up anyhow

        for(i = 0; i < (int)std::sqrt(Points.size()); i++){ 
            // Get the coordinates of two random images
            coordinates1 = (this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates();
            coordinates2 = (this->Points)[R.generate_int_uniform(0,(int)(this->Points).size()-1)]->get_coordinates();

            allDistances.push(Kmetric->calculate_distance(coordinates1, coordinates2));

            for(j = 0; j < (int)coordinates1.size(); j++){ // Get the distances of some points we don't need many since we are rounding up anyhow
                this->CenterMass[j] += (coordinates1[j] + coordinates2[j]) / numberOfPointsToCheck;
            }
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
        while ((int)(this->Clusters).size() < K) {
            sumOfSquaredDistances = 0;  // Reset the sum of squared distances
            minDistances.clear();       // Reset the vector of minimum distances
            minDistances.push_back(std::make_pair(0.0, 0));  // Initialize the first element of the vector

            for (i = 0; i < (int)(this->Points).size(); i++) {
                minDistance = std::numeric_limits<double>::max();  // Start with a large value

                centroid = this->Clusters[0]->get_centroid();  // Assume first centroid is the closest
                double distance = Kmetric->calculate_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates());
                
                // Find the closest centroid
                for (j = 1; j < (int)(this->Clusters).size(); j++) {
                    centroid = this->Clusters[j]->get_centroid();
                    double tempDistance = Kmetric->calculate_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates());
                    if (tempDistance < distance) {
                        distance = tempDistance;
                    }
                }

                // minDistance = distance / maxDistance;  // Normalize
                minDistance = distance;  // Dont  Normalize
                minDistanceSquared = minDistance * minDistance; 
                sumOfSquaredDistances += minDistanceSquared;
                minDistances.push_back(std::make_pair(sumOfSquaredDistances, (this->Points)[i]->get_number()));
            }

            // Select the next centroid
            randomDistance = R.generate_double_uniform(0, sumOfSquaredDistances);
            for (i = 0; i < (int)(minDistances).size() - 1; i++) { // Minus 1 cause the random number can't be bigger than sumOfSquaredDistances, i.e. the last element
                if (randomDistance > minDistances[i].first && randomDistance < minDistances[i + 1].first) {
                    pointsNumber = minDistances[i].second;
                    cluster = std::make_shared<Cluster>(this->Points[pointsNumber]);
                    this->Clusters.push_back(cluster);
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
            distance = Kmetric->calculate_distance(point->get_coordinates(), tempCentroid->get_coordinates()); // Calculate the distance from each centroid
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
        int i;
        std::shared_ptr<Cluster> nearestCluster;
        for(i = 0; i < (int)(this->Points).size(); i++){
            nearestCluster = get_nearest_cluster((this->Points)[i]); // Get the nearest centroid to the point
            nearestCluster->add_point((this->Points)[i]); // Add the point to the cluster
            // printf("Cluster with centroid id: %d has %d points\n",(nearestCluster->get_centroid())->get_number(), (int)nearestCluster->get_points().size());
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
                    dinstaceBetweenCentroids = Kmetric->calculate_distance((this->Clusters)[i]->get_centroid()->get_coordinates(), (this->Clusters)[j]->get_centroid()->get_coordinates());
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
        for(const auto& item : unassignedImages){
            nearestCluster = get_nearest_cluster(item);
            nearestCluster->add_point(item);
        }
    }

    void traditional_convergence_algorithm(AssignmentFunction assignment){
        int i, clustersConvergedCounter;
        int clusterNumberConvergenceTolerance = int((double)(this->Clusters).size() * NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE); // If at 80% of the clusters are converged then we have converged
        bool hasConverged = false;

        std::vector<double> oldDistanceDifferenceVector((this->Clusters).size(), 0.0); // I need to initialize this with the distance of each centroid from the (0, 0...0) point
        double distanceOfNewCentroidFromThePrevious;
        double distanceDifferenceAsMaxPercentageTolerance = this->MaxDist * DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE; //  If the change in the distance is less than 1% of the max distance between two points in our dataset then we have converged
        
        double changeOfDistanceDifferencePercentage;

        printf("Need at least %d clusters to converge\n", clusterNumberConvergenceTolerance);
        printf("Tolerance percentage: %f\n", distanceDifferenceAsMaxPercentageTolerance);

        std::shared_ptr<ImageVector> newCentroid;

        for(i = 0; i < (int)(this->Clusters).size(); i++){ // Initialize the oldDistanceDifference
            oldDistanceDifferenceVector[i] = Kmetric->calculate_distance((this->Clusters)[i]->get_centroid()->get_coordinates(), this->CenterMass);
        }

        printf("Initialized the old differences vector successfully\n");

        do{
            assignment(); // Assign the points to the clusters
            clustersConvergedCounter = 0;
            for(i = 0; i < (int)(this->Clusters).size(); i++){ // Recalculate the centroids
                newCentroid = (this->Clusters)[i]->recalculate_centroid();
                distanceOfNewCentroidFromThePrevious = Kmetric->calculate_distance(newCentroid->get_coordinates(), (this->Clusters)[i]->get_centroid()->get_coordinates());

                changeOfDistanceDifferencePercentage = (oldDistanceDifferenceVector[i] - distanceOfNewCentroidFromThePrevious) / oldDistanceDifferenceVector[i];
                
                oldDistanceDifferenceVector[i] = distanceOfNewCentroidFromThePrevious;

                printf("%d distance: %f and percentage change: %f\n", i, distanceOfNewCentroidFromThePrevious, changeOfDistanceDifferencePercentage);

                // Going from a step of 100 to a step of 5 is a 95% step change which means that we've basically found our converence point
                // But going from a step of 100 to a step of 50, means that our rate of convergence hasn't changed much and we could keep going
                // In any case, if we are taking steps that are very small in orders of magnitude (less than 1% of our max distance) then we can stop as well
                if(distanceOfNewCentroidFromThePrevious < distanceDifferenceAsMaxPercentageTolerance || changeOfDistanceDifferencePercentage >= CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE){ // If the distance is bigger than the tolerance
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

    void mac_queen_with_lloyds(){
        int i; 
        int epochs = 0;
        bool converged = false; // Initialize the convergence flag

        std::shared_ptr<Cluster> nearestCluster;
        // std::shared_ptr<Cluster> cluster;
        std::shared_ptr<ImageVector> centroid;

        do{
            printf("Epoch: %d\n", epochs++);
            // Save the previous centroids
            std::vector<std::shared_ptr<ImageVector>> previousCentroids;
            for(auto& cluster : this->Clusters){
                previousCentroids.push_back(std::make_shared<ImageVector>(-1, cluster->get_centroid()->get_coordinates()));
                cluster->get_points().clear(); // Clear the points of the cluster
                // printf("Image centroid has a positive number: %d\n", cluster->get_centroid()->get_number());
            }

            // printf("Got previous centroids\n");

            for (i = 0; i < (int)(this->Points.size()); i++){
                nearestCluster = get_nearest_cluster(this->Points[i]); 
                nearestCluster->add_point_and_set_centroid(this->Points[i]);
            }

            // printf("Assigned each point to the nearest cluster\n");

            // Check for convergence by comparing the new centroids with the previous centroids
            converged = true;
            for(i = 0; i < (int)(this->Clusters.size()); i++){
                double centroidDistance = Kmetric->calculate_distance(previousCentroids[i]->get_coordinates(), this->Clusters[i]->get_centroid()->get_coordinates());
                // printf("Centroid %d distance: %f\n", i, centroidDistance);
                converged = converged && (centroidDistance < DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE * (this->MaxDist));
            }
        }while(!converged);
    }

    void mac_queen_with_reverse(std::shared_ptr<ApproximateMethods> method){

        int  countRange, countConflicts;

        bool converged = false; // Initialize the convergence flag

        int i, j, convergedCounter;
        int epochs = 0;

        double radius;

        int clusterNumberConvergenceTolerance = int((double)(this->Clusters).size() * NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE); // If at 80% of the clusters are converged then we have converged

        std::vector<int> previousNumberOfPoints;

        std::shared_ptr<ImageVector> image;
        std::shared_ptr<Cluster> cluster;
        std::unordered_set<std::shared_ptr<ImageVector>> unassignedImages;
        std::vector<std::shared_ptr<ImageVector>> imagesToErase;

        std::vector<std::shared_ptr<Cluster>> prospectiveClusters;

        for(auto& image : this->Points){
            unassignedImages.insert(image);
        }

        // Find the minimum distance between centroids
        double minimumCentroidDistance = DBL_MAX;
        double tempDistance;
        for(i = 0; i < (int)this->Clusters.size(); i++){
            for(j = 0; j < i; j++){
                tempDistance  = Kmetric->calculate_distance(this->Clusters[i]->get_centroid()->get_coordinates(), this->Clusters[j]->get_centroid()->get_coordinates());
                if(tempDistance < minimumCentroidDistance){
                    minimumCentroidDistance = tempDistance;
                }
            }
        }

        // Start with radius = min(dist between centers)/2
        radius = minimumCentroidDistance / 2;
        
        do{
            printf("Epoch %d\n", epochs++);
            countConflicts = 0;
            countRange = 0;

            imagesToErase.clear();
            previousNumberOfPoints.clear();
            convergedCounter = 0;
            // Matching each image with its prospective clusters in every iteration
            std::map<std::shared_ptr<ImageVector>, std::vector<std::shared_ptr<Cluster>>> imagesToProspectiveClusters;
            
            // Mark the images that are inside the ranges
            for(i = 0; i < (int)(this->Clusters).size(); i++){
                cluster = (this->Clusters)[i];
                previousNumberOfPoints.push_back((int)cluster->get_points().size());

                // Get the in range images (which come in the form of pair<double, ImageVector>)
                auto distanceImagePairs = method->approximate_range_search_return_images(cluster->get_centroid(), radius);
                // auto distanceImagePairs = exhaustive_range_search(this->Points, cluster->get_centroid(), radius, this->Kmetric);
                for(auto& distanceImagePair : distanceImagePairs){
                    // Get the image from the pair
                    image = distanceImagePair.second;
                    // Mark assigned points: Push the cluster into the images prospective clusters
                    // If an image has no prospective clusters then it is not assigned
                    imagesToProspectiveClusters[image].push_back(cluster);
                }
            }
            // For a given radius, if a point lies in ≥ 2 balls, compare its distances to the respective centroids, assign to closest centroid.
            for(auto& image : unassignedImages){
                prospectiveClusters = imagesToProspectiveClusters[image];
                if((int)prospectiveClusters.size() == 1){
                    prospectiveClusters[0]->add_point_and_set_centroid(image);
                    imagesToErase.push_back(image);
                    countRange++;
                }
                else if((int)prospectiveClusters.size() > 1){
                    auto nearestCluster = get_nearest_cluster(image);
                    nearestCluster->add_point_and_set_centroid(image);
                    imagesToErase.push_back(image);
                    countConflicts++;
                }
            }

            printf("Range: %d Conflicts: %d\n", countRange, countConflicts);

            //Erase the flagged images
            for(auto& image : imagesToErase){
                unassignedImages.erase(image);
            }

            for(i = 0; i < (int)(this->Clusters).size(); i++){
                cluster = (this->Clusters)[i];
                if((int)cluster->get_points().size() == previousNumberOfPoints[i]){
                    convergedCounter++;
                }
            }
            // Until most balls get no new point.
            if(convergedCounter >= clusterNumberConvergenceTolerance){
                converged = true;
            }
            // Multiply radii by 2
            radius *= 2;
        }while(!converged); // Until most balls get no new point.

        // For every unassigned point, compare its distances to all centroids
        for(auto& image : unassignedImages){
            auto nearestCluster = get_nearest_cluster(image);
            nearestCluster->add_point(image);   // Don't set its centroid, we are done 
        }
    }

    std::shared_ptr<Cluster> get_second_nearest_cluster(std::shared_ptr<ImageVector> point){
        int j;
        double distance;
        std::shared_ptr<ImageVector> tempCentroid;
        std::shared_ptr<Cluster> nearestCluster;
        std::shared_ptr<Cluster> secondNearestCluster;
        std::priority_queue<std::pair<double, std::shared_ptr<Cluster>>, std::vector<std::pair<double, std::shared_ptr<Cluster>>>, std::greater<std::pair<double, std::shared_ptr<Cluster>>>> distancesFromCentroids;

        for(j = 0; j < (int)(this->Clusters).size(); j++){
            tempCentroid = (this->Clusters)[j]->get_centroid();
            distance = Kmetric->calculate_distance(point->get_coordinates(), tempCentroid->get_coordinates()); // Calculate the distance from each centroid
            distancesFromCentroids.push(make_pair(distance, (this->Clusters)[j]));
        }
        distancesFromCentroids.pop(); // Pop the first element which is the nearest cluster
        return  distancesFromCentroids.top().second;
    }

    double silhouette(){
        int numPoints = (int)this->Points.size();
        int numClusters = (int)(this->Clusters).size();

        // Since I've indexed the images beginning from 1, I need to add one more row and column to the matrix in order to not just -1 every single index
        std::vector<std::vector<double>> distancesMatrix(numPoints+1, std::vector<double>(numPoints+1, -1.0)); 

        double silhouette = 0;

        for(int i = 0; i < numClusters; i++){

            double cluster_silhouette = 0;

            int clusterSize = (int)(this->Clusters[i])->get_points().size();
            printf("Cluster %d size %d\n", i, clusterSize);
            fflush(stdout);

            for(int j = 0; j < clusterSize; j++){
                double object_silhouette;
                auto secondNearestCluster = get_second_nearest_cluster((this->Clusters[i])->get_points()[j]);

                double average_distance_on_same_cluster = 0;
                double average_distance_on_other_cluster = 0;

                int pointJNumber = (this->Clusters[i])->get_points()[j]->get_number();

                // printf("Got pointJNumber = %d\n", pointJNumber);

                // if(pointJNumber == -1){
                //     printf("You are doing something terribly wrong mate, check out the Clustering functions agian\n");
                //     return -1;
                // }

                for(int k = 0; k < clusterSize; k++){ // For this cluster
                    int pointKNumber = (this->Clusters[i])->get_points()[k]->get_number();

                    // printf("Got pointKNumber = %d\n", pointKNumber);

                    // if(pointKNumber == -1){
                    //     printf("You are doing something terribly wrong mate, check out the Clustering functions agian\n");
                    //     return -1;
                    // }

                    if(distancesMatrix[pointJNumber][pointKNumber] == -1){ 
                        double tempdistance = Kmetric->calculate_distance((this->Clusters[i])->get_points()[j]->get_coordinates(), 
                                                                        (this->Clusters[i])->get_points()[k]->get_coordinates());
                        distancesMatrix[pointJNumber][pointKNumber] = tempdistance;
                        distancesMatrix[pointKNumber][pointJNumber] = tempdistance;
                        // printf("Save new value distances[%d][%d] =  %f\n",pointJNumber, pointKNumber,  tempdistance);
                    }
                    average_distance_on_same_cluster += distancesMatrix[pointJNumber][pointKNumber];
                }

                int secondClusterSize = (int)(secondNearestCluster)->get_points().size();

                for(int l = 0; l < secondClusterSize; l++){ // For the other cluster
                    int pointLNumber = (secondNearestCluster)->get_points()[l]->get_number();
                    // printf("Got pointLNumber = %d\n", pointLNumber);

                    // if(pointLNumber == -1){
                    //     printf("You are doing something terribly wrong mate, check out the Clustering functions agian\n");
                    //     return -1;
                    // }

                    if(distancesMatrix[pointJNumber][pointLNumber] == -1){
                        double tempdistance = Kmetric->calculate_distance((this->Clusters[i])->get_points()[j]->get_coordinates(), 
                                                                        (secondNearestCluster)->get_points()[l]->get_coordinates());
                        distancesMatrix[pointJNumber][pointLNumber] = tempdistance;
                        distancesMatrix[pointLNumber][pointJNumber] = tempdistance;
                        // printf("Save new value distances[%d][%d] =  %f\n",pointJNumber, pointLNumber,  tempdistance);
                    }
                    average_distance_on_other_cluster += distancesMatrix[pointJNumber][pointLNumber];
                }

                // printf("Entering two divisions...");
                average_distance_on_same_cluster /= (double)clusterSize;
                average_distance_on_other_cluster /= (double)secondClusterSize;
                // printf("Done, okay\n");

                // printf("Calculating object silhouette\n");
                
                object_silhouette = (average_distance_on_other_cluster - average_distance_on_same_cluster) / 
                                    std::max(average_distance_on_other_cluster, average_distance_on_same_cluster);

                // printf("Object %d silhouette: %f\n", j, object_silhouette);
                    
                cluster_silhouette += object_silhouette;
            }
            cluster_silhouette /= (double)clusterSize;

            printf("Cluster %d silhouette: %f\n", i, cluster_silhouette);
            fflush(stdout);

            silhouette += cluster_silhouette;
        }

        silhouette /= (double)numClusters;
        return silhouette;
    }

};

int main(void){
    Eucledean metric;
    std::vector<std::shared_ptr<ImageVector>> dataset = read_mnist_images("./in/query.dat", 0);

    std::shared_ptr<kMeans> kmeans;
    kmeans = std::make_shared<kMeans>(10, dataset, &metric);

    double sumOfDistances = 0;
    double d = 0;

    printf("Number of centroids: %d\n", (int)kmeans->get_centroids().size());
    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){
        printf("Centroid %d: %d\n", i, kmeans->get_centroids()[i]->get_number());
    }

    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){ 
        sumOfDistances = 0;
        for(int j = 0; j < (int)kmeans->get_centroids().size(); j++){ 
            sumOfDistances += metric.calculate_distance(kmeans->get_centroids()[i]->get_coordinates(), kmeans->get_centroids()[j]->get_coordinates());
        }
        d += (sumOfDistances/(int)kmeans->get_centroids().size());
    }
    printf("Average Distance Between centroids: %f\n", d/10);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // kmeansInstance.mac_queen([&kmeansInstance](){ kmeansInstance.lloyds_assignment(); });
    // kmeans->traditional_convergence_algorithm([kmeans](){kmeans->lloyds_assigment();});
    
    // kmeansInstance.mac_queen(std::bind(&kMeans::reverse_assignment, &kmeansInstance, method));
    

    // Lloyd's
    // start = std::chrono::high_resolution_clock::now(); // Start the timer
    // kmeans->mac_queen_with_lloyds();
    // end  = std::chrono::high_resolution_clock::now(); // End the timer 


    std::shared_ptr<LSH> lsh;
    lsh = std::make_shared<LSH>(5,8, MODULO, LSH_TABLE_SIZE, &metric);
    lsh->load_data(dataset);

    // std::shared_ptr<HyperCube> cube;
    // cube = std::make_shared<HyperCube>(14, 500, 2000,&metric);
    // cube->load_data(dataset);


    // original one
    // start = std::chrono::high_resolution_clock::now(); // Start the timer
    // kmeans->traditional_convergence_algorithm(std::bind(&kMeans::reverse_assignment, kmeans, lsh));
    // end  = std::chrono::high_resolution_clock::now(); // End the timer 

    // The "correct" one that tries to resolve conflicts
    start = std::chrono::high_resolution_clock::now(); // Start the timer
    kmeans->mac_queen_with_reverse(lsh);
    end  = std::chrono::high_resolution_clock::now(); // End the timer 

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Time: %ld\nCalculating silhouette...\n",  duration);

    printf("%f\n", kmeans->silhouette());

    return 0;
}
