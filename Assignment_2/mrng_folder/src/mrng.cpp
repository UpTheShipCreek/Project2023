#include <vector>
#include <memory>
#include <unordered_set>
#include <queue>
#include <algorithm>

#include "graph.h"

class MonotonicRelativeNeighborGraph : public Graph{

    public: 
    // Give a set of nodes, i.e. the d dimensional points/images in our dataset 
    MonotonicRelativeNeighborGraph(
        std::vector<std::shared_ptr<ImageVector>> nodes,  
        Metric* metric) : 
        Graph(nodes, metric){ // Constructor

        double distance, leastDistance, edgepv;

        // Get the graph structure that maps the nodes to their neighbors
        // Apparently it needs to be specified reference in the type
        // std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>>& nodesNeighbors = get_nodes_neighbors();

        std::priority_queue<
            std::pair<double, std::shared_ptr<ImageVector>>,  // A priority queue of pairs of distance and node
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, // Saving it in a vector of pairs<double, ImageVector>
            std::greater<std::pair<double, std::shared_ptr<ImageVector>>> // We need the priority queue to be sorted in ascending order of the distance
        > sortedRp;
            
        // ----- Construction Process ----- //
        // Get the set of nodes so we can easily create some subsets
        // Don't do that btw, just do a loop inside the for, that creates the Rp and just doesn't include in it 
        std::unordered_set<std::shared_ptr<ImageVector>> setOfNodes(get_nodes().begin(), get_nodes().end()); 
        // for every node p in nodes 
        for(auto& p : nodes){
            // Create a set that doesn't contain the current node called Rp
            std::unordered_set<std::shared_ptr<ImageVector>> Rp(setOfNodes);
            Rp.erase(p);
            
            // Sort Rp according to the distance to the current node
            for(auto& node : Rp){
                distance = this->GraphMetric->calculate_distance(p->get_coordinates(), node->get_coordinates());
               
                // Maybe here we can also save the distance in a map so we can easily access it later
                sortedRp.push(std::make_pair(distance, node));
            }

            // Make a copy of the sortedRp since we are about to ruin it 
            std::priority_queue<
                std::pair<double, std::shared_ptr<ImageVector>>,
                std::vector<std::pair<double, std::shared_ptr<ImageVector>>>,
                std::greater<std::pair<double, std::shared_ptr<ImageVector>>>
            > copiedRp(sortedRp);

            // Initialize another set Lp which is comprised of all the nodes with minimum distance to p
            leastDistance = sortedRp.top().first;

            std::shared_ptr<Neighbors> Lp = std::make_shared<Neighbors>();
            while(copiedRp.top().first == leastDistance){

                // Add the node to Lp
                Lp->push_back(copiedRp.top().second);

                // Get the next element of the priority queue
                copiedRp.pop();

                // If the priority queue is empty, break
                if(copiedRp.empty()) break;
            }
            
            // for every node v in (Rp - Lp) 
            while(!sortedRp.empty()){
                auto& v = sortedRp.top().second;
                sortedRp.pop();
                if(std::find(Lp->begin(), Lp->end(), v) != Lp->end()) continue; 

                // and t in Lp (Lp is a shared_ptr or Neighbor type so we need to dereference it)
                for(auto& t : *Lp){
                    // if edge(p,v) is not the longest edge in the triangle (p, v, t) 
                    edgepv = this->GraphMetric->calculate_distance(p->get_coordinates(), v->get_coordinates());
                    if(
                        // edge(p,v) < edge(p,t)
                        edgepv < this->GraphMetric->calculate_distance(p->get_coordinates(), t->get_coordinates())
                        &&
                        // edge(p,v) < edge(v,t)
                        edgepv < this->GraphMetric->calculate_distance(v->get_coordinates(), t->get_coordinates())
                    ){
                        // Add r in Lp
                        Lp->push_back(v);
                    }
                }
            }
            // std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>>
            // Return Lp for all p in nodes, so edges are {edge(p,v) | p in nodes, v in Lp}
            // Fill up the graph structure
            this->NodesNeighbors[p] = Lp;
        }
    }
    
    // Need to implement this
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> k_nearest_neighbor_search(){

    }
};

int main(void){
    Eucledean metric;

    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> lsh_nearest_approx;
    
    // Get queries
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/query.dat", (int)images.size());

    std::vector<std::shared_ptr<ImageVector>> smallDataset;
    auto endIter = queries.begin();
    std::advance(endIter, std::min(1000, static_cast<int>(queries.size())));
    smallDataset.assign(queries.begin(), endIter);

    MonotonicRelativeNeighborGraph mrng(smallDataset, &metric);

    std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>> nodesNeighbors = mrng.get_nodes_neighbors();
    std::vector<std::shared_ptr<ImageVector>> nodes = mrng.get_nodes();

    if(nodesNeighbors.size() != smallDataset.size() || nodes.size() != smallDataset.size()){
        printf("Something went wrong\n");
        printf("Size of nodesNeighbors = %d and number of nodes = %d\n", (int)nodesNeighbors.size(), (int)nodes.size());
        return -1;
    }
    else{
        printf("The initialization seems to be working correctly\n");
    }


    LSH lsh(LSH_DEFAULT_L, LSH_DEFAULT_K, MODULO, LSH_TABLE_SIZE, &metric);
    lsh.load_data(smallDataset);

    for(int i = 1; i <= 10; i++){
        printf("Query: %d\n", i);

        int greedySteps = 10;
        nearest_approx = mrng.k_nearest_neighbor_search();
        lsh_nearest_approx = lsh.approximate_k_nearest_neighbors_return_images(images[i], GRAPH_DEFAULT_N);
        for(int i = 0; i < (int)nearest_approx.size(); i++){
            printf("LSH: %f Graph: %f\n", lsh_nearest_approx[i].first, nearest_approx[i].first);
        }
        printf("\n");
    }
}