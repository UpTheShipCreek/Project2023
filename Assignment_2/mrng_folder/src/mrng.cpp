#include <vector>
#include <memory>

#include "graph.h"

// q: For monotonic relative neighbor graph, are the nodes just the points in our d-dimensional space?



class MonotonicRelativeNeighborGraph : Graph{
    Metric* MRNGmetric; // The metric we are using to calculate the distance between nodes

    public: 
    // Give a set of nodes, i.e. the d dimensional points/images in our dataset 
    MonotonicRelativeNeighborGraph(std::vector<ImageVector> nodes, Metric* metric){ // Constructor
        
        this->MRNGmetric = metric;
        
        // ----- Construction Process ----- //
        // for every node p in nodes 
            // Create a set that doesn't contain the current node called Rp
            // Sort Rp according to the distance to the current node
            // Initialize another set Lp which is comprised of all the nodes with minimum distance to p
            // for every node v in (Rp - Lp) and t in Lp
                // if edge(p, v) is not the longest edge in the triangle (p, v, t) 
                // (edges don't normally have length but in that case we are using the distance between the nodes as the length)
                    // add v in Lp

        // Return Lp for all p in nodes, so edges are {edge(p,v) | p in nodes, v in Lp}
    }
};