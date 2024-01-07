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
    Random rand;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    auto originalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Read the original sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images("./in/input.dat", 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;
    if((int)dataset.size() != datasetHeaderInfo->get_numberOfImages()){
        printf("Dataset size does not match the header info (%d vs %d)\n", (int)dataset.size(), datasetHeaderInfo->get_numberOfImages());
        return -1;
    }

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> querysetInfo = read_mnist_images("./in/query.dat", (int)dataset.size());
    HeaderInfo* querysetHeaderInfo = querysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> queryset = querysetInfo.second;
    if((int)queryset.size() != querysetHeaderInfo->get_numberOfImages()){
        printf("Queryset size does not match the header info\n");
        return -1;
    }

    // Check that the shapes match between the two original sets
    if(!(*datasetHeaderInfo == *querysetHeaderInfo)){
        printf("Dataset and queryset shapes do not match\n");
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

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedQuerysetInfo = read_mnist_images("./in/encoded_queryset.dat", (int)reducedDataset.size());
    HeaderInfo* reducedQuerysetHeaderInfo = reducedQuerysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedQueryset = reducedQuerysetInfo.second;
    if((int)reducedQueryset.size() != reducedQuerysetHeaderInfo->get_numberOfImages()){
        printf("Reduced queryset size does not match the header info\n");
        return -1;
    }

    // Check that the shapes match between the two reduced sets
    if(!(*reducedDatasetHeaderInfo == *reducedQuerysetHeaderInfo)){
        printf("Reduced dataset and queryset shapes do not match\n");
        return -1;
    }
    int reducedDimensions = reducedDatasetHeaderInfo->get_numberOfRows() * reducedDatasetHeaderInfo->get_numberOfColumns();
    
    // Set up the correspondances, we can then use the number of the image to get the original coordinates with get_initial
    SpaceCorrespondace datasetSpaceCorrespondace(dataset);

    // Original Kmeans
    std::shared_ptr<kMeans> kmeans = std::make_shared<kMeans>(10, dataset, &metric);

    // Reduced Kmeans
    std::shared_ptr<kMeans> reducedKmeans = std::make_shared<kMeans>(10, reducedDataset, &metric);


    start = std::chrono::high_resolution_clock::now();
    kmeans->mac_queen_with_lloyds();
    end = std::chrono::high_resolution_clock::now();
    originalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    reducedKmeans->mac_queen_with_lloyds();
    end = std::chrono::high_resolution_clock::now();
    reducedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


    // Get the reduced clusters in order to traslate them to the original space
    std::vector<std::shared_ptr<Cluster>>& reducedClusters = reducedKmeans->get_clusters();
    printf("%d\n", __LINE__);
    fflush(stdout);

    std::vector<std::shared_ptr<Cluster>> translatedReducedClusters;
    std::shared_ptr<Cluster> translatedCluster;
    for(auto& cluster : reducedClusters){
        printf("%d\n", __LINE__);
        fflush(stdout);

        std::shared_ptr<ImageVector> centroid = cluster->get_centroid();
        std::vector<std::shared_ptr<ImageVector>> points = cluster->get_points();

        printf("%d\n", __LINE__);
        fflush(stdout);
        
        // Translate the images to the original space
        std::vector<std::shared_ptr<ImageVector>> translatedPoints;

        printf("%d\n", __LINE__);
        fflush(stdout);

        for(auto& point : points){
            std::shared_ptr<ImageVector> translated = datasetSpaceCorrespondace.get_initial(point->get_number());
            translatedPoints.push_back(translated);
        }
        printf("%d\n", __LINE__);
        fflush(stdout);
        // Then recaulculate the centroid
        translatedCluster = std::make_shared<Cluster>(translatedPoints);
        printf("%d\n", __LINE__);
        fflush(stdout);
        translatedCluster->set_centroid(translatedCluster->recalculate_centroid());

        printf("%d\n", __LINE__);
        fflush(stdout);

        // Push it back to our translated vector of clusters
        translatedReducedClusters.push_back(translatedCluster);
        printf("%d\n", __LINE__);
        fflush(stdout);

    }

    std::shared_ptr<kMeans> translatedKmeans = std::make_shared<kMeans>(translatedReducedClusters, &metric);

    // Now we can do the sihlouettes
    std::vector<double> originalSilhouettes = kmeans->silhouette();
    std::vector<double> reducedSilhouettes = translatedKmeans->silhouette();


    double cOT = originalTime.count() / 1e9;
    double cRT = reducedTime.count() / 1e9;

    printf("Original Time: %f", cOT);
    for(int i = 0; i < (int)originalSilhouettes.size(); i++){
        printf("Original Silhouette[%d]: %f\n", i, originalSilhouettes[i]);
    }
    printf("Reduced Time: %f", cRT);
    for(int i = 0; i < (int)reducedSilhouettes.size(); i++){
        printf("Reduced Silhouette[%d]: %f\n", i, reducedSilhouettes[i]);
    }

    return 0;
}
