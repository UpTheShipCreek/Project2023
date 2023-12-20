#include <stdio.h>
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "io_functions.h"
#include "mrng.h"

#define QUERY_LIMIT 10
#define GREEDY_STEPS 10

int main(void){

    Eucledean metric;

    std::string datasetFilename = "./in/input.dat";
    std::string querysetFilename = "./in/query.dat";

    // Read images from file
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images(datasetFilename, 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;
    
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> querysetInfo = read_mnist_images(querysetFilename, (int)dataset.size());
    HeaderInfo* querysetHeaderInfo = querysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> queryset = querysetInfo.second;

    if(!(*datasetHeaderInfo == *querysetHeaderInfo)){
        printf("Dataset and queryset header info do not match\n");
        return -1;
    }

    printf(
        "Dataset size: %d vs %d\nQueryset size: %d vs %d\nNumber of Bytes: %d\n", 
        (int)dataset.size(), datasetHeaderInfo->get_numberOfImages(),
        (int)queryset.size(), querysetHeaderInfo->get_numberOfImages(),
        datasetHeaderInfo->get_numberOfRows() * datasetHeaderInfo->get_numberOfColumns()
    );

    int dimensions = datasetHeaderInfo->get_numberOfRows() * datasetHeaderInfo->get_numberOfColumns();

    // LSH(int l, int k, double window, int tableSize, Metric* metric)
    std::shared_ptr<LSH> lsh = std::make_shared<LSH>(5, 4, 1400, 3750, &metric, dimensions);
    lsh->load_data(dataset);
    
    
    // // HyperCube(int dimensions, int probes, int numberOfElementsToCheck, Metric* metric)
    // std::shared_ptr<HyperCube> hypercube = std::make_shared<HyperCube>(11, 600, 4000, &metric);
    // hypercube->load_data(dataset);

    // // Generic Graph Search 
    // std::shared_ptr<Graph> genericGraph = std::make_shared<Graph>(dataset, &metric);
    // genericGraph->initialize_neighbours_approximate_method(lsh, 50);

    // // Monotonic Relative Neighbor Graph
    // std::shared_ptr<MonotonicRelativeNeighborGraph> monotonicGraph = std::make_shared<MonotonicRelativeNeighborGraph>(dataset, lsh, 400, &metric);



    return 0;
}
