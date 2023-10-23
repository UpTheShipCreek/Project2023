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
        long double numberOfPoints = (long double)((this->Points).size());
        long double fraction = (numberOfPoints) / (numberOfPoints + 1);

        long double temp;
        long double newvalue;


        for (int i = 0; i < (int)(this->Centroid)->get_coordinates().size(); i++){
            temp =  (long double)(this->Centroid)->get_coordinates()[i];
            newvalue = (fraction * temp) + (point->get_coordinates()[i] / (numberOfPoints + 1));
            
            (this->Centroid)->get_coordinates()[i] = (double)newvalue; // THIS IS THE BUG, GET COORDINATES DOESN'T RETURN THE VECTOR BUT A COPY OF IT, MASSIVE CHANGES INC I GUESS

            if(temp != newvalue){
                printf("%f -> %f %f\n", (double)temp, (double)newvalue, (this->Centroid)->get_coordinates()[i]);
            }
        }
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
                        distancesFromCentroids.push(Kmetric->calculate_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates()));
                    }
                }
                minDistance = distancesFromCentroids.top(); // Get the first element, which least distance from a centroid
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

        double distanceDifferenceTolerance = this->MaxDist * DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE;

        std::shared_ptr<Cluster> nearestCluster;
        // std::shared_ptr<Cluster> cluster;
        std::shared_ptr<ImageVector> centroid;

        do{
            printf("Epoch: %d\n", epochs++);
            // Save the previous centroids
            std::vector<std::shared_ptr<ImageVector>> previousCentroids;
            for(const auto& cluster : this->Clusters){
                previousCentroids.push_back(std::make_shared<ImageVector>(-1, cluster->get_centroid()->get_coordinates()));
                printf("Image centroid has a positive number: %d\n", cluster->get_centroid()->get_number());
            }

            printf("Got previous centroids\n");

            //for (i = 0; i < (int)(this->Points.size()); i++){
            for (i = 0; i < 10; i++){
                nearestCluster = get_nearest_cluster(this->Points[i]);
                
                // centroid = nearestCluster->get_centroid();
                // printf("\n[");
                // for(int j = 0; j < (int)centroid->get_coordinates().size(); j++){
                //     printf("%f ", centroid->get_coordinates()[j]);
                //     fflush(stdout);
                // }
                // printf("]\n");
               
                nearestCluster->add_point_and_set_centroid(this->Points[i]);
                
                // printf("\n[");
                // for(int j = 0; j < (int)centroid->get_coordinates().size(); j++){
                //     printf("%f ", centroid->get_coordinates()[j]);
                //     fflush(stdout);
                // }
                // printf("]\n");
            }

            printf("Assigned each point to the nearest cluster\n");

            // Check for convergence by comparing the new centroids with the previous centroids
            converged = true;
            for(i = 0; i < (int)(this->Clusters.size()); i++){
                double centroidDistance = Kmetric->calculate_distance(previousCentroids[i]->get_coordinates(), this->Clusters[i]->get_centroid()->get_coordinates());
                printf("Centroid %d distance: %f\n", i, centroidDistance);
                converged = converged && (centroidDistance > distanceDifferenceTolerance);
            }
        }while(!converged);
    }

    void mac_queen_with_reverse(std::shared_ptr<ApproximateMethods> method){
        bool converged = false;
        int i, j; 
        int epochs = 0;

        double dinstaceBetweenCentroids, radius;
        double minDistanceBetweenCentroids = DBL_MAX;
        double maxDistanceBetweenCentroids = DBL_MIN;

        double distanceDifferenceTolerance = this->MaxDist * DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE;


        std::shared_ptr<ImageVector> centroid;
        std::shared_ptr<Cluster> nearestCluster;
        // std::shared_ptr<Cluster> cluster;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> inRangeImages;

        
        // Create a new structure that keeps track of the assigned images
        std::shared_ptr<ImageVector> tempImageVector;
        std::unordered_set<std::shared_ptr<ImageVector>> unassignedImages;

        do{
            printf("Epoch: %d\n", epochs++);
            std::vector<std::shared_ptr<ImageVector>> previousCentroids;
            for(const auto& cluster : this->Clusters){
                previousCentroids.push_back(cluster->get_centroid());
            }

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
                centroid = nearestCluster->get_centroid();
                printf("\n[");
                for(int j = 0; j < (int)centroid->get_coordinates().size(); j++){
                    printf("%f ", centroid->get_coordinates()[j]);
                    fflush(stdout);
                }
                printf("]\n");
                nearestCluster->add_point_and_set_centroid(item);
                printf("\n[");
                for(int j = 0; j < (int)centroid->get_coordinates().size(); j++){
                    printf("%f ", centroid->get_coordinates()[j]);
                    fflush(stdout);
                }
                printf("]\n");
            }

            // Check for convergence by comparing the new centroids with the previous centroids
            converged = true; //  Need to reset this, otherwise we will never converge
            for(i = 0; i < (int)(this->Clusters.size()); i++){
                double centroidDistance = Kmetric->calculate_distance(previousCentroids[i]->get_coordinates(), this->Clusters[i]->get_centroid()->get_coordinates());
                printf("Centroid %d distance: %f\n", i, centroidDistance);
                converged = converged && (centroidDistance > distanceDifferenceTolerance);
            }
        }while(!converged);
    }
};

int main(void){
    Eucledean metric;
    std::vector<std::shared_ptr<ImageVector>> dataset = read_mnist_images("./in/query.dat", 0);

    std::shared_ptr<kMeans> kmeans;
    kmeans = std::make_shared<kMeans>(10, dataset, &metric);

    double sumOfDistances = 0;
    printf("Number of centroids: %d\n", (int)kmeans->get_centroids().size());
    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){
        printf("Centroid %d: %d\n", i, kmeans->get_centroids()[i]->get_number());
    }

    for(int i = 0; i < (int)kmeans->get_centroids().size(); i++){ 
        for(int j = 0; j < (int)kmeans->get_centroids().size(); j++){ 
            sumOfDistances += metric.calculate_distance(kmeans->get_centroids()[i]->get_coordinates(), kmeans->get_centroids()[j]->get_coordinates());
        }
    }
    printf("Metric: %f\n", sumOfDistances);

    std::shared_ptr<LSH> lsh;
    lsh = std::make_shared<LSH>(4,5,MODULO,LSH_TABLE_SIZE, &metric);
    lsh->load_data(dataset);


    // kmeansInstance.mac_queen([&kmeansInstance](){ kmeansInstance.lloyds_assignment(); });
    // kmeans->traditional_convergence_algorithm([kmeans](){kmeans->lloyds_assigment();});
    
    // kmeansInstance.mac_queen(std::bind(&kMeans::reverse_assignment, &kmeansInstance, method));
    // kmeans->traditional_convergence_algorithm(std::bind(&kMeans::reverse_assignment, kmeans, lsh));

    kmeans->mac_queen_with_lloyds();

    //kmeans->mac_queen_with_reverse(lsh);

    return 0;
}
