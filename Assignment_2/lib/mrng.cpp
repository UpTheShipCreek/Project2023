#include "mrng.h"

MonotonicRelativeNeighborGraph::MonotonicRelativeNeighborGraph(
    std::vector<std::shared_ptr<ImageVector>> nodes,  
    Metric* metric) : 
    Graph(nodes, metric){ // Constructor
    
    printf("Constructing MRNG Graph... ");
    fflush(stdout);
    int i;
    // double leastDistance;
    double distance, edgepv, edgept, edgevt;
    double newValue;
    double fraction;

    // A zero vector to initialize the centroid with
    std::vector<double> vectorZero(nodes[0]->get_coordinates().size(), 0.0);

    // Create a virtual point for the centroid
    this->Centroid = std::make_shared<ImageVector>(-1, vectorZero);

    std::shared_ptr<Neighbors> Lp;

    std::priority_queue<
        std::pair<double, std::shared_ptr<ImageVector>>,  // A priority queue of pairs of distance and node
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, // Saving it in a vector of pairs<double, ImageVector>
        std::greater<std::pair<double, std::shared_ptr<ImageVector>>> // We need the priority queue to be sorted in ascending order of the distance
    > sortedRp;
        
    // ----- Construction Process ----- //
    // For every node p in nodes 
    double nodeCount = 0;
    for(auto& p : nodes){

        // Incrementally building the centroid
        fraction = (nodeCount) / (nodeCount + 1);
        for (i = 0; i < (int)(this->Centroid)->get_coordinates().size(); i++){
            newValue = (fraction * (this->Centroid)->get_coordinates()[i]) + (p->get_coordinates()[i] / (nodeCount + 1));
            (this->Centroid)->get_coordinates()[i] = newValue;
        }
        nodeCount++;

        // Create a set that doesn't contain the current node called Rp
        // Sort Rp according to the distance to the current node
        for(auto& node : nodes){
            distance = this->GraphMetric->calculate_distance(p->get_coordinates(), node->get_coordinates());
            if(distance != 0.0) sortedRp.push(std::make_pair(distance, node));
            // Maybe here we can also save the distance in a map so we can easily access it later
        }

        Lp = std::make_shared<Neighbors>();
        Lp->push_back(sortedRp.top().second);
        
        // For every node v in Rp--
        bool flag;
        while(!sortedRp.empty()){
            flag = true;

            // Get the elements in sorted order
            auto& v = sortedRp.top().second;
            sortedRp.pop();
            
            // --and not in Lp
            if(std::find(Lp->begin(), Lp->end(), v) != Lp->end()) continue; 

            edgepv = this->GraphMetric->calculate_distance(p->get_coordinates(), v->get_coordinates());
            
            // and t in Lp (Lp is a shared_ptr or Neighbor type so we need to dereference it)
            for(auto& t : *Lp){
                // if edge(p,v) is NOT the longest edge in ANY triangle (p, v, t) i.e. it is shorter that at least one of the other edges
                edgept = this->GraphMetric->calculate_distance(p->get_coordinates(), t->get_coordinates());
                edgevt = this->GraphMetric->calculate_distance(v->get_coordinates(), t->get_coordinates());

                // if we find a triagle where edge(p,v) is the longest edge, then we break and return the false flag
                if(
                    // edge(p,v) < edge(p,t)
                    edgepv >= edgept
                    &&
                    // edge(p,v) < edge(v,t)
                    edgepv >= edgevt
                ){
                    flag = false;
                    break;
                }
            }
            if(flag){
                Lp->push_back(v);
            }
        }
        this->NodesNeighbors[p] = Lp;
    }

    // Find the closest real node to the virtual centroid of the dataset and assigns it to the NavigatingNode
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> vectorContainingNavigatingNode;
    vectorContainingNavigatingNode = exhaustive_nearest_neighbor_search_return_images(this->Nodes, this->Centroid, 1, this->GraphMetric);
    if (vectorContainingNavigatingNode.empty()){
        this->NavigatingNode = this->Nodes[this->RandGenerator.generate_int_uniform(0, (int)this->Nodes.size() - 1)];
    }
    else{
        this->NavigatingNode = vectorContainingNavigatingNode[0].second;
    }

    printf("Done\n");
    fflush(stdout);
}

//Calls the generic graph search with Navigating Node, which is the closest real node to the virtual centroid of the dataset, and returns it
std::vector<std::pair<double, std::shared_ptr<ImageVector>>> MonotonicRelativeNeighborGraph::k_nearest_neighbor_search(std::shared_ptr<ImageVector> query, int L, int K){
    return generic_k_nearest_neighbor_search(this->NavigatingNode, query, L, K);
}