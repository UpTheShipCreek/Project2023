#include <chrono>
#include "graph.h"


// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(void){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestApprox;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestExhaust;

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


    
    // dimensions: 11 probes: 600 numberOfPoints: 4000
    
    int const numberOfQueries = 1000;

    std::shared_ptr<HyperCube> cube;

    std::vector<int> mirrorLK = {4, 5, 6, 7};

    for(auto lk: mirrorLK){
        double maxFactor = DBL_MIN;
        double factor;
        double sum = 0;

        double approxSum = 0;
        double exhaustSum = 0;

        cube = std::make_shared<HyperCube>(11, 600, 4000, &metric);
        cube->load_data(images);

        for(int i = 0; i < numberOfQueries; i++){
            int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);


            start = std::chrono::high_resolution_clock::now();
            nearestApprox = cube->approximate_k_nearest_neighbors_return_images(queries[randomIndex], DEFAULT_N);
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
            sum += factor;
        }
        
        printf("LK: %d MaxFactor: %f AverageFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", lk, maxFactor, sum/numberOfQueries, approxSum / numberOfQueries, exhaustSum / numberOfQueries);
        fflush(stdout);
    }
}