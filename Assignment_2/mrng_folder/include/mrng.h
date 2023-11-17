#ifndef MRNG_H
#define MRNG_H 

#include "graph.h"

class MonotonicRelativeNeighborGraph : public Graph{
   std::shared_ptr<ImageVector> Centroid, NavigatingNode; // He navigate >:)

    public: 
    // Give a set of nodes, i.e. the d dimensional points/images in our dataset 
    MonotonicRelativeNeighborGraph(std::vector<std::shared_ptr<ImageVector>> nodes,  Metric* metric);

    //Calls the generic graph search with Navigating Node, which is the closest real node to the virtual centroid of the dataset, and returns it
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> k_nearest_neighbor_search(std::shared_ptr<ImageVector> query, int L, int K);
    
};

#endif