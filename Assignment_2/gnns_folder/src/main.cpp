#include <chrono>
#include "graph.h"


// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(void){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> true_nearest_approx;
    std::vector<std::shared_ptr<Neighbors>> nearest_approx_for_all;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = end - start;
    auto exhaustTime = end - start;

    Random rand;

    // std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

    // Get queries
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/query.dat", (int)queries.size());

    // LSH Optimal L: 6 K: 4 Window: 1400 TableSize: 7500
    std::shared_ptr<LSH> lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric);
    Graph genericGraph(images, &metric);
    genericGraph.initialize_neighbours_approximate_method(lsh, GRAPH_DEFAULT_K);
    int const numberOfQueries = 1000;

    std::vector<int> randomRestarts = {120, 130, 140, 150};
    for(auto r: randomRestarts){
        double maxFactor = DBL_MIN;
        double factor;
        double sum = 0;

        for(int i = 0; i < numberOfQueries; i++){
            int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);
            // GNNS Optimal K: 50 E: 30 G: 10 
            start = std::chrono::high_resolution_clock::now();
            nearest_approx = genericGraph.k_nearest_neighbor_search(queries[randomIndex], r, GRAPH_DEFAULT_G, GRAPH_DEFAULT_E, DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            approxTime = end - start;
            
            start = std::chrono::high_resolution_clock::now();
            true_nearest_approx = exhaustive_nearest_neighbor_search_return_images(images, queries[randomIndex], DEFAULT_N, &metric);
            end = std::chrono::high_resolution_clock::now();
            exhaustTime = end - start;

            factor = nearest_approx[0].first / true_nearest_approx[0].first;
            if(factor > maxFactor){
                maxFactor = factor;
            }
            sum += factor;
        }
        
        printf("RandomRestarts:%d MaxFactor:%f AverageFactor: %f ApproxAverage: %ld ExhaustAverage: %ld\n", r, maxFactor, sum/numberOfQueries, approxTime.count() / numberOfQueries, exhaustTime.count() / numberOfQueries);
        fflush(stdout);
    }
}