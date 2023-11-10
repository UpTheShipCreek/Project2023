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

// α) πληθος των πλησιεστερων γειτονων στον γραμμο k-NN 
// β) ακεραια παραμετρος Ε των επεκτασεων 
// γ) ακεραιος αριθμος R των τυχαιων επανεκκινησεων
// δ) ακεραιος αριμθος N των πλησιεστερων γειτονων
// (default: k=50, E=30, R=1, N=1)

#define GRAPH_DEFAULT_K 20 // default number of nearest neighbors that will be on the index
#define GRAPH_DEFAULT_E 10 // default number of expansions
#define GRAPH_DEFAULT_R 10 // default number of random restarts
#define GRAPH_DEFAULT_N 10 // default number of nearest neighbors we want to find


// FOR LSH
#define LSH_DEFAULT_L 5 // default number of hash tables
#define LSH_DEFAULT_K 4 // default number of hash functions


using Neighbors = std::vector<std::shared_ptr<ImageVector>>; 

class Graph{
    
    protected:
    std::vector<std::shared_ptr<ImageVector>> Nodes; 
    std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>> NodesNeighbors; // The list of neighbors for each node
    Random RandGenerator; // The random number generator we are using
    Metric* GraphMetric; // The metric we are using to calculate the distance between nodes

    // Note: Depending on how we initilaize the neighbor list it can we sorted or not
    // but initializing it with LSH/Hypercube will yield sorted results
    public:
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
        int L);
};

#endif