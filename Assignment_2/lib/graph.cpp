#include "graph.h"

const std::vector<std::shared_ptr<ImageVector>>& Graph::get_nodes(){
    return this->Nodes;
}

const std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Neighbors>>& Graph::get_nodes_neighbors(){
    return this->NodesNeighbors;
}

Graph::Graph(std::vector<std::shared_ptr<ImageVector>> nodes, Metric* metric){
    this->Nodes = nodes;
    this->GraphMetric = metric;  
}

Graph::Graph(std::vector<std::shared_ptr<ImageVector>> nodes, std::vector<std::shared_ptr<Neighbors>> neighborList, Metric* metric){
    printf("%d\n",__LINE__);
    this->Nodes = nodes;
    this->GraphMetric = metric;
    for(int i = 0; i < (int)nodes.size(); i++){
        this->NodesNeighbors[nodes[i]] = neighborList[i];
    }   
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> Graph::k_nearest_neighbor_search(
    std::shared_ptr<ImageVector> query, 
    int randomRestarts, int greedySteps, int expansions, int K){ 
    // expansions means the number of neighbors the N(Y,E,G) function, from the notes, will return I think
    printf("%d\n",__LINE__);
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
        std::less<std::pair<double, std::shared_ptr<ImageVector>>> // We need the top element in the PQ to be the one with the largest distance so we can quickly remove it
    > S;

    std::unordered_set<int> priorityQueueNodeNumbers; // This could prove problematic if we don't take care of the way we are loading multiple data files
    printf("%d\n",__LINE__);
    for(i = 0; i < randomRestarts; i++){
        randomInt = RandGenerator.generate_int_uniform(0, (int)Nodes.size() - 1);
        printf("Number of nodes is %d and random number is %d\n",(int)Nodes.size(), randomInt);
        fflush(stdout);
        // Starting with a random node chosen uniformly 
        node = this->Nodes[randomInt];
        printf("The number of the random node is %d\n", node->get_number());
        // Replace current node Y_t-1 by the neighbor that is closest to the query
        for(j = 0; j < greedySteps; j++){
            // S = S /union N(Y_t-1, E, G)
            // So, we need to get the neighbors of the current node
            // Then keep the E closest
            // And add them to our set S, which is a priority queue atm
            // In order to maintain our S structure as a priority Queue
            // We will keep track of the set of the node ids that are in the priority queue
            printf("%d\n",__LINE__);
            // N(Y_t-1, E, G), i.e. keep the first E neighbors of the node
            neighbors = this->NodesNeighbors[node];
            printf("Number of neighbors of %d node is %d\n", node->get_number(), (int)neighbors->size());
            auto neighborsKeepE = Neighbors(neighbors->begin(), neighbors->begin() + expansions);
            printf("%d\n",__LINE__);
            for(auto tempNode : neighborsKeepE){
                // Calcuate the distance of the neighbor to the query
                distance = GraphMetric->calculate_distance(tempNode->get_coordinates(), query->get_coordinates());
                if(distance < minDistance){
                    minDistance = distance;
                    minDistanceNode = tempNode;
                }
                printf("%d\n",__LINE__);
                nodesIndexNumber = tempNode->get_number();
                // If the neighbor is not already in the priority queue
                if(priorityQueueNodeNumbers.find(nodesIndexNumber) == priorityQueueNodeNumbers.end()){
                    printf("%d\n",__LINE__);
                    // Add the neighbor to the priority queue
                    S.push(std::make_pair(distance, tempNode));
                    printf("%d\n",__LINE__);
                    if((int)S.size() > K){
                        S.pop();
                    }
                    printf("%d\n",__LINE__);
                    priorityQueueNodeNumbers.insert(nodesIndexNumber);
                }
            }
            // Yt = minD(Y,Q) for Y in nearest neighbors of Yt-1
            printf("%d\n",__LINE__);
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




// L is the number of candidates we are allowed to check, similar to M in hypercube
// K is the number of nearest neighbors we want to find
// Returns a vector of pairs of the distance and the imagevector/node
// Broken, can't iterate through a PQ without breaking into a protected structure of the container, so I'll need another solution
// Maybe sort in the end instead of using a priority queue, or use a set as well as the priority queue, but then I'll need 
// To manage both
std::vector<std::pair<double, std::shared_ptr<ImageVector>>> Graph::generic_k_nearest_neighbor_search(std::shared_ptr<ImageVector> startNode, std::shared_ptr<ImageVector> query, int L){
    int i;
    double distance;

    std::shared_ptr<ImageVector> node;
    std::shared_ptr<Neighbors> neighbors;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;

    // Initialize Candiate set, which will be actually a priority queue since we need it to be sorted
    // std::priority_queue<
    //     std::pair<double, std::shared_ptr<ImageVector>>,  // A priority queue of pairs of distance and node
    //     std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, // Saving it in a vector of pairs<double, ImageVector>
    //     std::less<std::pair<double, std::shared_ptr<ImageVector>>> // We need the priority queue to be sorted in ascending order of the distance
    // > R;


    std::vector<std::shared_ptr<ImageVector>> R;   
    std::vector<double> vectorOfDistances;
    std::map<double, std::shared_ptr<ImageVector>> distancesToNodes;

    // MAYBE MAKE A MAP OF DISTANCES TO NODES, SO WE CAN PUSH THE NODES IN THE PRIORITY QUEUE 
    // THEN BEING SORTED BY DISTANCE AND THEN HAVE SAID DISTANCE ASSOCIATED WITH AN ELEMENT 

    distance = this->GraphMetric->calculate_distance(startNode->get_coordinates(), query->get_coordinates());

    // From now on you need to handle all of those structures together
    R.push_back(startNode);
    vectorOfDistances.push_back(distance);
    distancesToNodes[distance] = startNode;
    

    // Initialize the set of unchecked nodes
    std::unordered_set<std::shared_ptr<ImageVector>> checkedNodes;

    // while i < L
    i = 1;
    while(i < L){
        // For every node in the candidates 
        bool foundUncheckedCandidate = false;
        for(auto& node : R){
            // See if we have already checked this node
            if(checkedNodes.find(node) == checkedNodes.end()){
                // If we haven't checked this node, continue the checking process with it
                checkedNodes.insert(node);
                foundUncheckedCandidate = true;
                break;
            }
        }

        // If we failed to find an unchecked candidate node, we are done
        if(!foundUncheckedCandidate) break;

        // Get the neighbors of the candidate node
        neighbors = NodesNeighbors[node];

        // for all neighbors of the candidate node 
        for(auto& neighbor : *neighbors){
            // if neighbor is not in the candidate set, i.e. not in R, add it 
            if(std::find(R.begin(), R.end(), neighbor) != R.end()){

                // Add the neighbor to the candidate set
                distance = this->GraphMetric->calculate_distance(neighbor->get_coordinates(), query->get_coordinates());
                R.push_back(neighbor);
                vectorOfDistances.push_back(distance);
                distancesToNodes[distance] = neighbor;

                // If we have exceeded the number of nearest neighbors we are allowed to check, remove the farthest neighbor
                i++;
            }
        }
    }
    // sort R in ascending order of the distance to q
    std::sort(vectorOfDistances.begin(), vectorOfDistances.end());
    for(auto& distance : vectorOfDistances){
        nearestImages.push_back(std::make_pair(distance, distancesToNodes[distance]));
    }
    return nearestImages;
}