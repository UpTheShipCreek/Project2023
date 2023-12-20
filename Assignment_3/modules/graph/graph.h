#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <memory>
#include <map>
#include <unordered_set>
#include <queue>
#include <cfloat>  // For MAX_DOUBLE

#include "image_util.h"
#include "random_functions.h"
#include "metrics.h"
#include "lsh.h"
#include "hypercube.h"
#include "approximate_methods.h"

using Neighbors = std::vector<std::shared_ptr<ImageVector>>; 

class Graph{
    
    protected:
    std::vector<std::shared_ptr<ImageVector>> Nodes; 
    std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>> NodesNeighbors; // The list of neighbors for each node

    // Note: Depending on how we initilaize the neighbor list it can we sorted or not
    // but initializing it with LSH/Hypercube will yield sorted results
    public:
    Random RandGenerator; // The random number generator we are using
    Metric* GraphMetric; // The metric we are using to calculate the distance between nodes
    Graph(std::vector<std::shared_ptr<ImageVector>> nodes, Metric* metric);
    Graph(std::vector<std::shared_ptr<ImageVector>> nodes, std::vector<std::shared_ptr<Neighbors>> neighborList, Metric* metric);

    const std::vector<std::shared_ptr<ImageVector>>& get_nodes();
    const std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>>& get_nodes_neighbors();

    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> k_nearest_neighbor_search(
        std::shared_ptr<ImageVector> query, 
        int randomRestarts, int greedySteps, int expansions, int K);

    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> generic_k_nearest_neighbor_search(
        std::shared_ptr<ImageVector> startNode, 
        std::shared_ptr<ImageVector> query, 
        int L, int K);

    void initialize_neighbours_approximate_method(std::shared_ptr<ApproximateMethods> method, int k);
};

#endif