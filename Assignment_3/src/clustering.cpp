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

int main(int argc, char **argv){

    Eucledean metric;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    auto originalClusteringTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedClusteringTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto originalSilhouetteTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedSilhouetteTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);


    std::string inputFileName, reducedInputFileName;

    if(argc != 3){
        printf("Error: Argument Number. Example call: ./clustering <original dataset> <reduced dataset>\n");
        return -1;
    }
    inputFileName = argv[1];
    reducedInputFileName = argv[2];

    // Read the original sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images(inputFileName, 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;
    if(dataset.empty()){
        printf("Error reading file: %s\n", inputFileName.c_str());
        return -1;
    }
    if((int)dataset.size() != datasetHeaderInfo->get_numberOfImages()){
        printf("Dataset size does not match the header info (%d vs %d)\n", (int)dataset.size(), datasetHeaderInfo->get_numberOfImages());
        return -1;
    }

    // Read the reduced sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedDatasetInfo = read_mnist_images(reducedInputFileName, 0);
    HeaderInfo* reducedDatasetHeaderInfo = reducedDatasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedDataset = reducedDatasetInfo.second;
    if(reducedDataset.empty()){
        printf("Error reading file: %s\n", reducedInputFileName.c_str());
        return -1;
    }
    if((int)reducedDataset.size() != reducedDatasetHeaderInfo->get_numberOfImages()){
        printf("Reduced dataset size does not match the header info\n");
        return -1;
    }
    
    // Set up the correspondances, we can then use the number of the image to get the original coordinates with get_initial
    SpaceCorrespondace datasetSpaceCorrespondace(dataset);

    // Original Kmeas
    std::shared_ptr<kMeans> kmeans = std::make_shared<kMeans>(DEFAULT_NUMBER_OF_CLUSTERS, dataset, &metric);

    // Reduced Kmeans
    std::shared_ptr<kMeans> reducedKmeans = std::make_shared<kMeans>(DEFAULT_NUMBER_OF_CLUSTERS, reducedDataset, &metric);

    int sizeOfOriginalSpace = (int)dataset[0]->get_coordinates().size();

    start = std::chrono::high_resolution_clock::now();
    kmeans->mac_queen_with_lloyds();
    end = std::chrono::high_resolution_clock::now();
    originalClusteringTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double oCT = originalClusteringTime.count() / 1e9;
    printf("Original Clustering Time: %f\n", oCT);

    start = std::chrono::high_resolution_clock::now();
    reducedKmeans->mac_queen_with_lloyds();
    end = std::chrono::high_resolution_clock::now();
    reducedClusteringTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double rCT = reducedClusteringTime.count() / 1e9;
    printf("Reduced Clustering Time: %f\n", rCT);


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

    // Silhouettes for the original clustering
    std::vector<double> originalSilhouettes = kmeans->silhouette();
    // Print the results for the original dimension clustering
    for(int i = 0; i < (int)originalSilhouettes.size()-1; i++){
        printf("Original Silhouette[%d]: %f\n", i, originalSilhouettes[i]);
    }
    printf("Average Silhouette: %f\n", originalSilhouettes[originalSilhouettes.size()-1]);
    
    // Silhouettes for the reduced clustering
    std::vector<double> reducedSilhouettes = translatedKmeans->silhouette();
    // Print the results for the reduced dimension clustering
    for(int i = 0; i < (int)reducedSilhouettes.size()-1; i++){
        printf("Reduced Silhouette[%d]: %f\n", i, reducedSilhouettes[i]);
    }
    printf("Average Silhouette: %f\n", reducedSilhouettes[reducedSilhouettes.size()-1]);

    return 0;
}
