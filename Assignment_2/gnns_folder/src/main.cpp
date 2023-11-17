#include <chrono>
#include <iostream>
#include "graph.h"


// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(void){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestApprox;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestExhaust;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto exhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    Random rand;

    // std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

    // Get queries
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/query.dat", (int)queries.size());

    int const retries = 100;
    int const numberOfQueries = 10;
    int const billion = std::pow(10, 9);

    // L: 6 K: 4 Window: 1400 TableSize: 7500
    std::shared_ptr<LSH> lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric);

    // dimensions: 11 probes: 600 numberOfPoints: 4000
    std::shared_ptr<HyperCube> cube = std::make_shared<HyperCube>(11, 600, 4000, &metric);

    // #define GRAPH_DEFAULT_K 50 // default number of nearest neighbors that will be on the index
    // #define GRAPH_DEFAULT_E 30 // default number of expansions
    // #define GRAPH_DEFAULT_R 120 // default number of random restarts
    // #define GRAPH_DEFAULT_G 10 // default number of greedy steps
    Graph genericGraph(images, &metric);

    int mode;
    printf("Press anything other than 1:");
    fflush(stdout);
    std::cin >> mode;
    if(mode == 1){
        std::vector<int> kappas = {100};
        std::vector<int> es = {100}; 
        std::vector<int> restarts = {200, 300, 400};
        std::vector<int> greedySteps = {10, 20, 30};

        for(auto k : kappas){
            genericGraph.initialize_neighbours_approximate_method(lsh, k);
            for(auto e : es){
                for (auto r : restarts){
                    for(auto g : greedySteps){
                        double maxFactor = DBL_MIN;
                        double factor;
                        double sum = 0;

                        double approxSum = 0;
                        double exhaustSum = 0;

                        for(int j = 0; j < retries; j++){
                            for(int i = 0; i < numberOfQueries; i++){
                                int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);


                                start = std::chrono::high_resolution_clock::now();
                                nearestApprox = genericGraph.k_nearest_neighbor_search(queries[randomIndex], r, g, e, DEFAULT_N);
                                if(nearestApprox.size() == 0){
                                    printf("Error: No nearest neighbors found\n");
                                    break;
                                }
                                end = std::chrono::high_resolution_clock::now();
                                end = std::chrono::high_resolution_clock::now();
                                approxTime = end - start;
                                
                                start = std::chrono::high_resolution_clock::now();
                                nearestExhaust = exhaustive_nearest_neighbor_search_return_images(images, queries[randomIndex], DEFAULT_N, &metric);
                                end = std::chrono::high_resolution_clock::now();
                                exhaustTime = end - start;

                                factor = nearestApprox[0].first / nearestExhaust[0].first;
                                if(factor > maxFactor){
                                    maxFactor = factor;
                                }
                                approxSum += approxTime.count();
                                exhaustSum += exhaustTime.count();
                                
                            }
                            sum += maxFactor;
                        }
                        double tA = approxSum / (numberOfQueries*retries);
                        double tE = exhaustSum / (numberOfQueries*retries);
                        printf("K: %d Restarts: %d Greedy: %d Expansions:%d AverageMaxFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", k, r, g, e, sum/retries, tA / billion, tE / billion);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    // K: 100 Restarts: 400 Greedy: 10 Expansions:100 AverageMaxFactor: 2.527122 ApproxAverage: 0.005116 ExhaustAverage: 0.046603
    else{
        genericGraph.initialize_neighbours_approximate_method(lsh, 100);

        std::vector<int> queryNumber = {1000, 2000, 3000, 4000, 5000, 10000};
    
        for(auto q : queryNumber){

            double maxFactor = DBL_MIN;
            double factor;
            double approxSum = 0;
            double exhaustSum = 0;

            for(int i = 0; i < q; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);


                start = std::chrono::high_resolution_clock::now();
                nearestApprox = genericGraph.k_nearest_neighbor_search(queries[randomIndex], 400, 10, 100, DEFAULT_N);
                if(nearestApprox.size() == 0){
                    printf("Error: No nearest neighbors found\n");
                    break;
                }
                end = std::chrono::high_resolution_clock::now();
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;
                
                start = std::chrono::high_resolution_clock::now();
                nearestExhaust = exhaustive_nearest_neighbor_search_return_images(images, queries[randomIndex], DEFAULT_N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                factor = nearestApprox[0].first / nearestExhaust[0].first;
                if(factor > maxFactor){
                    maxFactor = factor;
                }
                approxSum += approxTime.count();
                exhaustSum += exhaustTime.count();
            }
            double tA = approxSum / q;
            double tE = exhaustSum / q;
            printf("Number of Queries:%d MaxFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", q, maxFactor, tA/billion, tE/billion);
            fflush(stdout);
        }
    }
}