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
    std::vector<std::shared_ptr<ImageVector>> Nodes; // The nodes of the graph
    std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>> NodesNeighbors; // The list of neighbors for each node
    Metric* GraphMetric; // The metric we are using to calculate the distance between nodes
    Random RandGenerator; // The random number generator we are using

    // Note: Depending on how we initilaize the neighbor list it can we sorted or not
    // but initializing it with LSH/Hypercube will yield sorted results
    public:
    Graph(std::vector<std::shared_ptr<ImageVector>> nodes, std::vector<std::shared_ptr<Neighbors>> neighborList, Metric* metric){
        // printf("%d\n",__LINE__);
        this->Nodes = nodes;
        this->GraphMetric = metric;
        for(int i = 0; i < (int)nodes.size(); i++){
            this->NodesNeighbors[nodes[i]] = neighborList[i];
        }   
        // printf("%d\n",__LINE__);
    }

    // L is the number of candidates we are allowed to check, similar to M in hypercube
    // K is the number of nearest neighbors we want to find
    // Returns a vector of pairs of the distance and the imagevector/node
    // Broken, can't iterate through a PQ without breaking into a protected structure of the container, so I'll need another solution
    // Maybe sort in the end instead of using a priority queue, or use a set as well as the priority queue, but then I'll need 
    // To manage both
    // std::vector<std::pair<double, ImageVector>> generic_k_nearest_neighbor_search(std::shared_ptr<ImageVector> startNode, std::shared_ptr<ImageVector> query, int L, int K){
    //     int i;
    //     double distance;

    //     std::shared_ptr<ImageVector> node;
    //     std::shared_ptr<Neighbors> neighbors;
    //     std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;

    //     // Initialize Candiate set, which will be actually a priority queue since we need it to be sorted
    //     std::priority_queue<
    //         std::pair<double, std::shared_ptr<ImageVector>>,  // A priority queue of pairs of distance and node
    //         std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, // Saving it in a vector of pairs<double, ImageVector>
    //         std::less<std::pair<double, std::shared_ptr<ImageVector>>> // We need the priority queue to be sorted in ascending order of the distance
    //     > R;

    //     distance = this->GraphMetric->calculate_distance(startNode->get_coordinates(), query->get_coordinates());
    //     R.push(std::make_pair(distance, startNode));

    //     // Initialize the set of unchecked nodes
    //     std::unordered_set<std::shared_ptr<ImageVector>> checkedNodes;

    //     // while i < L
    //     i = 1;
    //     while(i < L){
    //         // For every node in the candidates 
    //         bool foundUncheckedCandidate = false;
    //         for(auto& distanceNodePair : R){
    //             node = distanceNodePair.second;
    //             // See if we have already checked this node
    //             if(checkedNodes.find(node) == checkedNodes.end()){
    //                 // If we haven't checked this node, continue the checking process with it
    //                 checkedNodes.insert(node);
    //                 foundUncheckedCandidate = true;
    //                 break;
    //             }
    //         }

    //         // If we failed to find an unchecked candidate node, we are done
    //         if(!foundUncheckedCandidate) break;

    //         // Get the neighbors of the candidate node
    //         neighbors = NodesNeighbors[node];

    //         // for all neighbors of the candidate node 
    //         for(auto& neighbor : neighbors){
    //             // if neighbor is not in the candidate set, i.e. not in R, add it 
    //             if(R.find(neighbor) == R.end()){

    //                 // Add the neighbor to the candidate set
    //                 dinstace = this->GraphMetric->distance(neighbor, query);
    //                 R.push(std::make_pair<distance, neighbor>);

    //                 // If we have exceeded the number of nearest neighbors we are allowed to check, remove the farthest neighbor
    //                 if((int)R.size() > K){
    //                     R.pop();
    //                 }
    //                 i++;
    //             }
    //         }
    //         // "sort R in ascending order of the distance to q", which we don't need to do cause we are using a Priority Queue
    //         // We do need return a vector of the K nearest neighbors in reverse though
    //         while (!R.empty()){
    //             nearestImages.push_back(R.top());
    //             R.pop();
    //         }
    //         std::vector<std::pair<double, ImageVector>> reversed(nearestImages.rbegin(), nearestImages.rend());
    //         return reversed;
    //     }
    // }
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> k_nearest_neighbor_search(
        std::shared_ptr<ImageVector> query, 
        int randomRestarts, int greedySteps, int expansions, int K){ 
        // expansions means the number of neighbors the N(Y,E,G) function, from the notes, will return I think
        // printf("%d\n",__LINE__);
        // The number of expansions can't be greater that the number of neighbors we want to find
        if(expansions > K) expansions = K;

        int i, j, randomInt, nodesIndexNumber;
        
        double distance;
        double minDistance = DBL_MAX;
        
        std::shared_ptr<ImageVector> node;
        std::shared_ptr<ImageVector> minDistanceNode;
        std::shared_ptr<Neighbors> neighbors;

        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;
        
        std::priority_queue<
            std::pair<double, std::shared_ptr<ImageVector>>,  // A priority queue of pairs of distance and node
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, // Saving it in a vector of pairs<double, ImageVector>
            std::less<std::pair<double, std::shared_ptr<ImageVector>>> // We need the priority queue to be sorted in ascending order of the distance
        > S;

        std::unordered_set<int> priorityQueueNodeNumbers; // This could prove problematic if we don't take care of the way we are loading multiple data files
        // printf("%d\n",__LINE__);
        for(i = 0; i < randomRestarts; i++){
            randomInt = RandGenerator.generate_int_uniform(0, (int)Nodes.size() - 1);
            // printf("%d\n",__LINE__);
            fflush(stdout);
            // Starting with a random node chosen uniformly 
            node = Nodes[randomInt];
            // printf("%d\n",__LINE__);
            // Replace current node Y_t-1 by the neighbor that is closest to the query
            for(j = 0; j < greedySteps; j++){
                // S = S /union N(Y_t-1, E, G)
                // So, we need to get the neighbors of the current node
                // Then keep the E closest
                // And add them to our set S, which is a priority queue atm
                // In order to maintain our S structure as a priority Queue
                // We will keep track of the set of the node ids that are in the priority queue
                // printf("%d\n",__LINE__);
                // N(Y_t-1, E, G), i.e. keep the first E neighbors of the node
                neighbors = NodesNeighbors[node];
                auto neighborsKeepE = Neighbors(neighbors->begin(), neighbors->begin() + expansions);
                // printf("%d\n",__LINE__);
                for(auto tempNode : neighborsKeepE){
                    // Calcuate the distance of the neighbor to the query
                    distance = GraphMetric->calculate_distance(tempNode->get_coordinates(), query->get_coordinates());
                    if(distance < minDistance){
                        minDistance = distance;
                        minDistanceNode = tempNode;
                    }
                    // printf("%d\n",__LINE__);
                    nodesIndexNumber = tempNode->get_number();
                    // If the neighbor is not already in the priority queue
                    if(priorityQueueNodeNumbers.find(nodesIndexNumber) == priorityQueueNodeNumbers.end()){
                        // printf("%d\n",__LINE__);
                        // Add the neighbor to the priority queue
                        S.push(std::make_pair(distance, tempNode));
                        // printf("%d\n",__LINE__);
                        if((int)S.size() > K){
                            S.pop();
                        }
                        // printf("%d\n",__LINE__);
                        priorityQueueNodeNumbers.insert(nodesIndexNumber);
                    }
                }
                // Yt = minD(Y,Q) for Y in nearest neighbors of Yt-1
                // printf("%d\n",__LINE__);
                node = minDistanceNode;
            }
        }

        // Reverse and Return the priority queue as a vector
        while (!S.empty()){
            nearestImages.push_back(S.top());
            S.pop();
        }
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> reversed(nearestImages.rbegin(), nearestImages.rend());
        return reversed;
    }
};

int main(void){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> lsh_nearest_approx;
    std::vector<std::shared_ptr<Neighbors>> nearest_approx_for_all;

    std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

    // Get queries
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/query.dat", (int)queries.size());


    // Load them into LSH
    LSH lsh(LSH_DEFAULT_L, LSH_DEFAULT_K, MODULO, LSH_TABLE_SIZE, &metric);
    lsh.load_data(images);
    printf("Creating the edge relations between the nodes/images...");
    fflush(stdout);
    for(auto& image : images){
        nearest_approx = lsh.approximate_k_nearest_neighbors_return_images(image, GRAPH_DEFAULT_K);
        for(auto& neighbor : nearest_approx){
            neighbors->push_back(neighbor.second);
        }
        nearest_approx_for_all.push_back(neighbors);
    }

    printf("Done\nCreating the graph...");
    fflush(stdout);
    Graph genericGraph(images, nearest_approx_for_all, &metric);
    printf("Done\nFinding nearest neighbors to the queries...\n");
    fflush(stdout);

    for(int i = 1; i <= 10; i++){
        printf("Query: %d\n", i);

        int greedySteps = 10;
        nearest_approx = genericGraph.k_nearest_neighbor_search(queries[i], GRAPH_DEFAULT_R, greedySteps, GRAPH_DEFAULT_E, GRAPH_DEFAULT_N);
        lsh_nearest_approx = lsh.approximate_k_nearest_neighbors_return_images(queries[i], GRAPH_DEFAULT_N);
        for(int i = 0; i < (int)nearest_approx.size(); i++){
            printf("LSH: %f Graph: %f\n", lsh_nearest_approx[i].first, nearest_approx[i].first);
        }
        printf("\n");
    }
    printf("Done\n");
    fflush(stdout);
}

