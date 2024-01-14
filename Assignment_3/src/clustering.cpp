#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <getopt.h>
#include <string.h>

#include "random_functions.h"
#include "io_functions.h"
#include "kmeans.h"

#define DEFAULT_NUMBER_OF_CLUSTERS 10


int main(void){

    Eucledean metric;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    auto originalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto originalSilhouetteTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedSilhouetteTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Read the original sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images("./in/input.dat", 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;
    if((int)dataset.size() != datasetHeaderInfo->get_numberOfImages()){
        printf("Dataset size does not match the header info (%d vs %d)\n", (int)dataset.size(), datasetHeaderInfo->get_numberOfImages());
        return -1;
    }


    // Read the reduced sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedDatasetInfo = read_mnist_images("./in/encoded_dataset.dat", 0);
    HeaderInfo* reducedDatasetHeaderInfo = reducedDatasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedDataset = reducedDatasetInfo.second;
    if((int)reducedDataset.size() != reducedDatasetHeaderInfo->get_numberOfImages()){
        printf("Reduced dataset size does not match the header info\n");
        return -1;
    }
    
    // Set up the correspondances, we can then use the number of the image to get the original coordinates with get_initial
    SpaceCorrespondace datasetSpaceCorrespondace(dataset);
    double originalOFR = 0.0;
    double reducedOFR = 0.0;
    for (int i = 0; i < 100; i++){
        // Original Kmeas
        // std::shared_ptr<kMeans> kmeans = std::make_shared<kMeans>(10, dataset, &metric);

        // Reduced Kmeans
        std::shared_ptr<kMeans> reducedKmeans = std::make_shared<kMeans>(10, reducedDataset, &metric);

        int sizeOfOriginalSpace = (int)dataset[0]->get_coordinates().size();
        // printf("Size of original space: %d\n", sizeOfOriginalSpace);

        // start = std::chrono::high_resolution_clock::now();
        // kmeans->mac_queen_with_lloyds();
        // end = std::chrono::high_resolution_clock::now();
        // originalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        // printf("Original time: %f\n", originalTime.count()/1e9);

        start = std::chrono::high_resolution_clock::now();
        reducedKmeans->mac_queen_with_lloyds();
        end = std::chrono::high_resolution_clock::now();
        reducedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("Reduced time: %f\n", reducedTime.count()/1e9);


        // Get the reduced clusters in order to traslate them to the original space
        std::vector<std::shared_ptr<Cluster>> reducedClusters = reducedKmeans->get_clusters();

        std::vector<std::shared_ptr<Cluster>> translatedReducedClusters;
        std::shared_ptr<Cluster> translatedCluster;
        std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Cluster>> pointToClusterMap;

        for(auto& cluster : reducedClusters){

            std::vector<std::shared_ptr<ImageVector>> points = cluster->get_points();
            

            // Make a new centroid in order to give it to a cluster pointer
            // Need first need to make a vector of coordinates
            std::vector<double> centroidCoordinates(sizeOfOriginalSpace, 0.0);
            std::shared_ptr<ImageVector> centroid = std::make_shared<ImageVector>(-1, centroidCoordinates);
            std::shared_ptr<Cluster> translatedCluster = std::make_shared<Cluster>(centroid);
        
            std::vector<std::shared_ptr<ImageVector>> translatedPoints;

            // Translate the images to the original space
            int countPoints = 0;
            std::shared_ptr<ImageVector> translated;
            for(auto& point : points){
                countPoints++;
                // printf("Image number: %d\n", point->get_number());
                translated = datasetSpaceCorrespondace.get_initial(point->get_number());
                pointToClusterMap[translated] = translatedCluster;
                translatedCluster->add_point_and_set_centroid(translated);
            }
            
            // Push it back to our translated vector of clusters
            translatedReducedClusters.push_back(translatedCluster);
        }

        std::shared_ptr<kMeans> translatedKmeans = std::make_shared<kMeans>(translatedReducedClusters, pointToClusterMap, &metric);
        
        // originalOFR += kmeans->get_objective_function_value();
        reducedOFR += translatedKmeans->get_objective_function_value();
    }
    // Square the mean
    // originalOFR /= 100.0;
    // originalOFR *= originalOFR;
    // printf("Original OFR: %f\n", originalOFR);

    reducedOFR /= 100.0;
    reducedOFR *= reducedOFR;
    printf("Reduced OFR: %f\n", reducedOFR);
    
    return 0;
}
