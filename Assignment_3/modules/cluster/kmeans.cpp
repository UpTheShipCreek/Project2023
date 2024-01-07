#include "kmeans.h"

double round_up_to_nearest_order_of_magnitude(double number){
    double order = std::pow(10, std::floor(std::log10(number)));
    return ceil(number / order) * order;
}

kMeans::kMeans(std::vector<std::shared_ptr<Cluster>> Clusters, std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Cluster>> PointToClusterMap, Metric* metric){
    this->Clusters = Clusters;
    this->K = (int)(this->Clusters).size();
    this->Kmetric = metric;
    this->PointToClusterMap = PointToClusterMap;
    
    for(int i = 0; i < (int)(this->Clusters).size(); i++){
        this->Points.insert(this->Points.end(), (this->Clusters)[i]->get_points().begin(), (this->Clusters)[i]->get_points().end());
    }
}

kMeans::kMeans(int k, std::vector<std::shared_ptr<ImageVector>> points, Metric* metric){ // Needs the number of clusters and the dataset
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

    // Assign the first centroid at random
    int firstCentroidIndex = R.generate_int_uniform(0, (int)(this->Points).size() - 1); // Get a random index for the first centroid
    std::shared_ptr<ImageVector> firstCentroid = this->Points[firstCentroidIndex]; // Get the first centroid

    cluster = std::make_shared<Cluster>(firstCentroid); // Make a cluster with the first centroid
    this->Clusters.push_back(cluster); // Save it in our clusters vector

    printf("Assigning Centroids... ");
    fflush(stdout);

    // Assign the rest of the centroids
    while ((int)(this->Clusters).size() < K){
        sumOfSquaredDistances = 0;  // Reset the sum of squared distances
        minDistances.clear();       // Reset the vector of minimum distances
        minDistances.push_back(std::make_pair(0.0, 0));  // Initialize the first element of the vector

        for (i = 0; i < (int)(this->Points).size(); i++){
            minDistance = std::numeric_limits<double>::max();  // Start with a large value

            centroid = this->Clusters[0]->get_centroid();  // Assume first centroid is the closest
            double distance = Kmetric->calculate_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates());
            
            // Find the closest centroid
            for (j = 1; j < (int)(this->Clusters).size(); j++){
                centroid = this->Clusters[j]->get_centroid();
                double tempDistance = Kmetric->calculate_distance((this->Points)[i]->get_coordinates(), centroid->get_coordinates());
                if (tempDistance < distance){
                    distance = tempDistance;
                }
            }

            minDistance = distance / maxDistance;  // Normalize
            // minDistance = distance;  // Dont  Normalize
            minDistanceSquared = minDistance * minDistance; 
            sumOfSquaredDistances += minDistanceSquared;
            minDistances.push_back(std::make_pair(sumOfSquaredDistances, (this->Points)[i]->get_number()));
        }

        // Select the next centroid
        randomDistance = R.generate_double_uniform(0, sumOfSquaredDistances);
        for (i = 0; i < (int)(minDistances).size() - 1; i++){ // Minus 1 cause the random number can't be bigger than sumOfSquaredDistances, i.e. the last element
            if (randomDistance > minDistances[i].first && randomDistance < minDistances[i + 1].first){
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
std::shared_ptr<Cluster> kMeans::get_nearest_cluster(std::shared_ptr<ImageVector> point){
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
std::vector<std::shared_ptr<ImageVector>> kMeans::get_centroids(){
    std::vector<std::shared_ptr<ImageVector>> centroids;
    for(int i = 0; i < (int)(this->Clusters).size(); i++){
        centroids.push_back((this->Clusters)[i]->get_centroid());
    }
    return centroids;
}
std::vector<std::shared_ptr<Cluster>>& kMeans::get_clusters(){
    return this->Clusters;
}

void kMeans::lloyds_assigment(){ // Lloyds-type assignment
    int i;
    std::shared_ptr<Cluster> nearestCluster;
    for(i = 0; i < (int)(this->Points).size(); i++){
        nearestCluster = get_nearest_cluster((this->Points)[i]); // Get the nearest centroid to the point
        nearestCluster->add_point((this->Points)[i]); // Add the point to the cluster
        // printf("Cluster with centroid id: %d has %d points\n",(nearestCluster->get_centroid())->get_number(), (int)nearestCluster->get_points().size());
    }
}
void kMeans::reverse_assignment(std::shared_ptr<ApproximateMethods> method){
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
    radius = minDistanceBetweenCentroids;
    while(!unassignedImages.empty() && radius < maxDistanceBetweenCentroids){ // Keep going until you have assigned all the points or the radius is bigger than the max distance between centroids
        // Assign imagevectors to clusters
        for(i = 0; i < (int)Clusters.size(); i++){
            (this->Clusters)[i]->get_points().clear(); // Clear the points of the cluster
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
void kMeans::traditional_convergence_algorithm(AssignmentFunction assignment){
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

            std::vector<double> previousCentroidCoordinates = (this->Clusters)[i]->get_centroid()->get_coordinates();
            newCentroid = (this->Clusters)[i]->recalculate_centroid(); 
            distanceOfNewCentroidFromThePrevious = Kmetric->calculate_distance(newCentroid->get_coordinates(), previousCentroidCoordinates);

            // The next two lines don't work anymore cause I am returning the actual reference to the cluster instead of a copy, 
            // so they will always have the coordinates
            // newCentroid = (this->Clusters)[i]->recalculate_centroid(); 
            // distanceOfNewCentroidFromThePrevious = Kmetric->calculate_distance(newCentroid->get_coordinates(), (this->Clusters)[i]->get_centroid()->get_coordinates());

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

void kMeans::mac_queen_with_lloyds(){
    printf("Clustering... ");
    fflush(stdout);
    int i; 
    bool converged = false; // Initialize the convergence flag

    std::shared_ptr<Cluster> nearestCluster;
    // std::shared_ptr<Cluster> cluster;
    std::shared_ptr<ImageVector> centroid;
    do{
        // Save the previous centroids
        std::vector<std::shared_ptr<ImageVector>> previousCentroids;
        for(auto& cluster : this->Clusters){
            previousCentroids.push_back(std::make_shared<ImageVector>(-1, cluster->get_centroid()->get_coordinates()));
        }

        // printf("Got previous centroids\n");

        for (i = 0; i < (int)(this->Points.size()); i++){
            nearestCluster = get_nearest_cluster(this->Points[i]); 

            if(this->PointToClusterMap.find(this->Points[i]) != this->PointToClusterMap.end()){ // If the point is already assigned to a cluster
                this->PointToClusterMap[this->Points[i]]->remove_point_and_set_centroid(this->Points[i]); // Remove it from the cluster
            }

            nearestCluster->add_point_and_set_centroid(this->Points[i]);
            this->PointToClusterMap[this->Points[i]] = nearestCluster; // This should end up saving the assigned cluster of each point
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
    printf("Done\n");
    fflush(stdout);
}
void kMeans::mac_queen_with_reverse(std::shared_ptr<ApproximateMethods> method){
    printf("Clustering... ");
    fflush(stdout);


    bool converged = false;

    int i, j, countRange, countConflicts;
    int epoch = 0;
    double radius, tempDistance, numberOfPointsChangedCluster, minimumCentroidDistance = DBL_MAX;
    
    std::shared_ptr<ImageVector> centroid;
    std::shared_ptr<ImageVector> image;
    std::shared_ptr<Cluster> nearestCluster;
    std::vector<std::shared_ptr<Cluster>> prospectiveClusters;
    std::unordered_set<std::shared_ptr<ImageVector>> unassignedImages;

    // Find the minimum distance between centroids
    for(i = 0; i < (int)this->Clusters.size(); i++){
        for(j = 0; j < i; j++){
            tempDistance  = Kmetric->calculate_distance(this->Clusters[i]->get_centroid()->get_coordinates(), this->Clusters[j]->get_centroid()->get_coordinates());
            if(tempDistance < minimumCentroidDistance){
                minimumCentroidDistance = tempDistance;
            }
        }
    }

    // Create a set with all the unassigned images, initialized as all the images
    for(auto& image : this->Points){
        unassignedImages.insert(image);
    }

    radius = minimumCentroidDistance;
    do{
        epoch++;
        numberOfPointsChangedCluster = 0;
        countConflicts = 0; 
        countRange = 0;
        std::map<std::shared_ptr<ImageVector>, std::vector<std::shared_ptr<Cluster>>> imagesToProspectiveClusters;

        for(auto& cluster : this->Clusters){
            centroid = cluster->get_centroid();
            // The approximate range search
            auto distanceImagePairs =  method->approximate_range_search_return_images(centroid, radius);
            
            // For the images that are in range, save the cluster as a prospective cluster
            for(auto& distanceImagePair : distanceImagePairs){
                // Get the image from the pair
                image = distanceImagePair.second;
                // Mark assigned points: Push the cluster into the images prospective clusters
                // If an image has no prospective clusters then it is not assigned
                imagesToProspectiveClusters[image].push_back(cluster);
            }
        }
        
        // For each image
        for(auto& image : this->Points){
            // Get the prospective clusters of the image
            prospectiveClusters = imagesToProspectiveClusters[image];

            if((int)prospectiveClusters.size() == 1){
                // If the new prospective cluster is different than the one that the image was assigned to
                if(this->PointToClusterMap[image] != prospectiveClusters[0]){
                    // Add the image to the new cluster
                    prospectiveClusters[0]->add_point_and_set_centroid(image);
                    // Delete it from the old one if it had one
                    if(this->PointToClusterMap[image] != nullptr){
                        this->PointToClusterMap[image]->remove_point_and_set_centroid(image);
                    }
                    // Update the map
                    this->PointToClusterMap[image] =  prospectiveClusters[0];
                    // Update the image counter of images that changed cluster for the convergence condition
                    numberOfPointsChangedCluster++;
                }
                unassignedImages.erase(image);
                countRange++;
            }
            else if((int)prospectiveClusters.size() > 1){ // If the image has multiple prospective clusters
                auto nearestCluster = get_nearest_cluster(image);
                // If the new cluster is different than the one that the image was assigned to
                if(this->PointToClusterMap[image] != nearestCluster){
                    // Add the image to the new cluster
                    nearestCluster->add_point_and_set_centroid(image);
                    // Delete it from the old one if it had one
                    if(this->PointToClusterMap[image] != nullptr){
                        this->PointToClusterMap[image]->remove_point_and_set_centroid(image);
                    }
                    // Update the map
                    this->PointToClusterMap[image] =  nearestCluster;
                    // Update the image counter of images that changed cluster for the convergence condition
                    numberOfPointsChangedCluster++;
                }
                unassignedImages.erase(image);
                countConflicts++;
            }
        }
        if(epoch >= LEAST_NUMBER_OF_EPOCHS && (numberOfPointsChangedCluster < (int)((double)this->Points.size() * OSCILATION_TOLERANCE) || radius > this->MaxDist)){
           converged = true;
        }
        // Multiply radii by 2
        radius *= 2;
    }while(!converged);

    for(auto& image : unassignedImages){
        auto nearestCluster = get_nearest_cluster(image);
        nearestCluster->add_point(image);   // We are assuming those are edge cases so there is no need to update centroids here

        this->PointToClusterMap[image] =  nearestCluster;
    }
    printf("Done\n");
    fflush(stdout);
}

std::shared_ptr<Cluster> kMeans::get_nearest_cluster_excluding_the_assigned_one(std::shared_ptr<ImageVector> point){
    int j;
    double distance;
    double minDistance = DBL_MAX;
    std::shared_ptr<ImageVector> tempCentroid;
    std::shared_ptr<Cluster> nearestCluster;

    std::shared_ptr<ImageVector> alreadyAssignedClusterCentroid = PointToClusterMap[point]->get_centroid();

    for(j = 0; j < (int)(this->Clusters).size(); j++){
        if(alreadyAssignedClusterCentroid == (this->Clusters)[j]->get_centroid()) continue; // If the cluster is the one that the point is already assigned to, skip it
        tempCentroid = (this->Clusters)[j]->get_centroid();
        distance = Kmetric->calculate_distance(point->get_coordinates(), tempCentroid->get_coordinates()); // Calculate the distance from each centroid
        if(distance < minDistance){
            minDistance = distance; // Get the minimum distance
            nearestCluster = (this->Clusters)[j]; // Get the nearest cluster
        }
    }
    return nearestCluster;
}
std::vector<double> kMeans::silhouette(){
    printf("Calculating Silhouette... ");
    fflush(stdout);

    int numPoints = (int)this->Points.size();
    int numClusters = (int)(this->Clusters).size();
    double maxAB;

    // Since I've indexed the images beginning from 1, I need to add one more row and column to the matrix in order to not just -1 every single index
    std::vector<std::vector<double>> distancesMatrix(numPoints+1, std::vector<double>(numPoints+1, -1.0)); 

    double silhouette = 0;
    std::vector<double> silhouetteVector;

    for(int i = 0; i < numClusters; i++){
        // printf("%d %s\n", __LINE__, __FUNCTION__);

        double cluster_silhouette = 0;

        int clusterSize = (int)(this->Clusters[i])->get_points().size();
        for(int j = 0; j < clusterSize; j++){
            // printf("%d %s\n", __LINE__, __FUNCTION__);
            double object_silhouette;
            auto secondNearestCluster = get_nearest_cluster_excluding_the_assigned_one((this->Clusters[i])->get_points()[j]);

            // printf("%d %s\n", __LINE__, __FUNCTION__);
            fflush(stdout);

            double average_distance_on_same_cluster = 0;
            double average_distance_on_other_cluster = 0;

            int pointJNumber = (this->Clusters[i])->get_points()[j]->get_number();
            // printf("%d %s\n", __LINE__, __FUNCTION__);
            fflush(stdout);

            for(int k = 0; k < clusterSize; k++){ // For this cluster
                // printf("%d %s\n", __LINE__, __FUNCTION__);
                int pointKNumber = (this->Clusters[i])->get_points()[k]->get_number();

                if(distancesMatrix[pointJNumber][pointKNumber] == -1){ 
                    double tempdistance = Kmetric->calculate_distance((this->Clusters[i])->get_points()[j]->get_coordinates(), 
                                                                    (this->Clusters[i])->get_points()[k]->get_coordinates());
                    distancesMatrix[pointJNumber][pointKNumber] = tempdistance;
                    distancesMatrix[pointKNumber][pointJNumber] = tempdistance;
                }
                average_distance_on_same_cluster += distancesMatrix[pointJNumber][pointKNumber];
            }

            int secondClusterSize = (int)(secondNearestCluster)->get_points().size();

            for(int l = 0; l < secondClusterSize; l++){ // For the other cluster
                // printf("%d %s\n", __LINE__, __FUNCTION__);
                int pointLNumber = (secondNearestCluster)->get_points()[l]->get_number();

                if(distancesMatrix[pointJNumber][pointLNumber] == -1){
                    double tempdistance = Kmetric->calculate_distance((this->Clusters[i])->get_points()[j]->get_coordinates(), 
                                                                    (secondNearestCluster)->get_points()[l]->get_coordinates());
                    distancesMatrix[pointJNumber][pointLNumber] = tempdistance;
                    distancesMatrix[pointLNumber][pointJNumber] = tempdistance;
                }
                average_distance_on_other_cluster += distancesMatrix[pointJNumber][pointLNumber];
            }
            // printf("%d %s\n", __LINE__, __FUNCTION__);

            average_distance_on_same_cluster /= (double)clusterSize;
            average_distance_on_other_cluster /= (double)secondClusterSize;
            
            maxAB = std::max(average_distance_on_other_cluster, average_distance_on_same_cluster);

            if(maxAB == 0.0) object_silhouette = 0.0;
            else object_silhouette = (average_distance_on_other_cluster - average_distance_on_same_cluster) / maxAB;
                
            cluster_silhouette += object_silhouette;
            silhouette += object_silhouette;
        }
        // printf("%d %s\n", __LINE__, __FUNCTION__);
        cluster_silhouette /= (double)clusterSize;
        silhouetteVector.push_back(cluster_silhouette);

    }
    // printf("%d %s\n", __LINE__, __FUNCTION__);
    silhouette /= numPoints;
    silhouetteVector.push_back(silhouette);

    printf("Done\n");
    fflush(stdout);
    return  silhouetteVector;
}