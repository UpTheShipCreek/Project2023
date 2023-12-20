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
    this->Nodes = nodes;
    this->GraphMetric = metric;
    for(int i = 0; i < (int)nodes.size(); i++){
        this->NodesNeighbors[nodes[i]] = neighborList[i];
    }   
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> Graph::k_nearest_neighbor_search(
    std::shared_ptr<ImageVector> query, 
    int randomRestarts, int greedySteps, int expansions, int K){ 
    // expansions means the number of neighbors the N(Y,E,G) function, from the notes, will return

    int i, j, randomInt, nodesIndexNumber;
    
    double distance;

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
    for(i = 0; i < randomRestarts; i++){
        randomInt = RandGenerator.generate_int_uniform(0, (int)Nodes.size() - 1);
        
        // Starting with a random node chosen uniformly 
        node = this->Nodes[randomInt];


        double previousMinDistance = DBL_MAX;

        // Replace current node Y_t-1 by the neighbor that is closest to the query
        for(j = 0; j < greedySteps; j++){

            // If the node has no neighbors, skip it
            if(this->NodesNeighbors[node]->size() == 0) break;

            // Else continue with this node
            neighbors = this->NodesNeighbors[node];
            
            // Don't exceed the number of neighbors we have available
            if(expansions > (int)neighbors->size()){
                expansions = (int)neighbors->size();
            }

            // Get the first E neighbors
            auto neighborsKeepE = Neighbors(neighbors->begin(), neighbors->begin() + expansions);
            
            double minDistance = DBL_MAX;
            minDistanceNode = nullptr;

            for(auto tempNode : neighborsKeepE){
                // Calcuate the distance of the neighbor to the query
                distance = GraphMetric->calculate_distance(tempNode->get_coordinates(), query->get_coordinates());
                if(distance < minDistance){
                    minDistance = distance;
                    minDistanceNode = tempNode;
                }
                
                // Get the nodes's number in order to check if it is already in the priority queue
                nodesIndexNumber = tempNode->get_number();

                // If the neighbor is not already in the priority queue
                if(priorityQueueNodeNumbers.find(nodesIndexNumber) == priorityQueueNodeNumbers.end()){
            
                    // Add the neighbor to the priority queue
                    S.push(std::make_pair(distance, tempNode));
                    
                    // Make sure you keep the size correct
                    if((int)S.size() > K){
                        S.pop();
                    }
                    
                    priorityQueueNodeNumbers.insert(nodesIndexNumber);
                }
            }

            // If we have converged at a local minimum then break
            if (minDistance >= previousMinDistance) {
                break;
            }

            // Update the previous minimum distance
            previousMinDistance = minDistance;
            
            // Just to be on the safe side
            if(minDistanceNode == nullptr){
                break;
            }
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

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> Graph::generic_k_nearest_neighbor_search(std::shared_ptr<ImageVector> startNode, std::shared_ptr<ImageVector> query, int L, int K){
    
    // Initializations
    bool foundUncheckedCandidate;
    
    double distance;

    std::shared_ptr<ImageVector> node;
    std::shared_ptr<Neighbors> neighbors;

    // The set of candidates we have already checked
    std::unordered_set<std::shared_ptr<ImageVector>> checkedCandidates; 
    
    // The set of candidates we are going to check
    std::vector<std::shared_ptr<ImageVector>> candidateSetR; 

    std::priority_queue<
        std::pair<double, std::shared_ptr<ImageVector>>, 
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>>,
        std::less<std::pair<double, std::shared_ptr<ImageVector>>>
    > sortedCandidateSetR;

    // "Add the starting node to the candidate set R"
    candidateSetR.push_back(startNode);

    // In our case it is a priority queue, so we will also add the node with its distance to the query
    distance = GraphMetric->calculate_distance(startNode->get_coordinates(), query->get_coordinates());
    sortedCandidateSetR.push(std::make_pair(distance, startNode));

    // For a certain amount of candidates L
    int i = 0;
    while(i < L){

        // ---------------- Finding an unchecked candidate ---------------- // 
        // For the candidates of set R
        foundUncheckedCandidate = false;
        for(auto& candidate : candidateSetR){
            // If a candidate is unchecked
            if(checkedCandidates.find(candidate) == checkedCandidates.end()){
                // Add the candidate to the set of checked candidates
                checkedCandidates.insert(candidate);
                // Don't forget to flip the flag
                foundUncheckedCandidate = true;
                // And save the candidate (apparently the iterator goes to the end)
                node = candidate;
                break;
            }
        }
        if(!foundUncheckedCandidate){
            // If we didn't find an unchecked candidate, break
            break;
        }
        neighbors = this->NodesNeighbors[node];
        for(auto& neighbor : *neighbors){
            // If the neighbor is not in the candidate set R
            if(std::find(candidateSetR.begin(), candidateSetR.end(), neighbor) == candidateSetR.end()){
                // Add the neighbor to the candidate set R
                candidateSetR.push_back(neighbor);
                // And add the neighbor to the sorted candidate set R
                distance = GraphMetric->calculate_distance(neighbor->get_coordinates(), query->get_coordinates());
                sortedCandidateSetR.push(std::make_pair(distance, neighbor));
                // But don't exceed the number of neighbors we want to return
                if((int)sortedCandidateSetR.size() > K){
                    sortedCandidateSetR.pop();
                }
            }
        }
        i++;
    }

    // Reverse and Return the priority queue as a vector
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;
    while (!sortedCandidateSetR.empty()){
        nearestImages.push_back(sortedCandidateSetR.top());
        sortedCandidateSetR.pop();
    }
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> reversed(nearestImages.rbegin(), nearestImages.rend());
    return reversed;
}

void Graph::initialize_neighbours_approximate_method(std::shared_ptr<ApproximateMethods> method, int k){
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;

    // Load the data into the approximate method
    method->load_data(this->Nodes);

    printf("Creating the edge relations between the nodes/images... ");
    fflush(stdout);

    for(auto& node : this->Nodes){
        nearest_approx = method->approximate_k_nearest_neighbors_return_images(node, k);
        
        // If the node has no neighbors, find the real ones with exhaustive search
        if(nearest_approx.empty()){
            nearest_approx = exhaustive_nearest_neighbor_search_return_images(this->Nodes, node, k, this->GraphMetric);
        }

        // Make sure you create different pointers for each "Neighbors" structure
        std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>(); 
        for(auto& neighbor : nearest_approx){
            neighbors->push_back(neighbor.second);
        }
        NodesNeighbors[node] = neighbors;
    }
    printf("Done\n");
    fflush(stdout);
}