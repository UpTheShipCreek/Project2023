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

    std::shared_ptr<HyperCube> cube;
    // HyperCube(int dimensions, int probes, int numberOfElementsToCheck, Metric* metric);
    int mode;
    printf("Press anything other than 1:");
    std::cin >> mode;
    if(mode == 1){
        std::vector<int> dimensions = {14};
        std::vector<int> probes = {500}; // Maybe there is a bug on hypercube's implementation if the probes are too many with regard to the dimensions
        std::vector<int> numberOfElementsToCheck = {1000,2000,3000,4000,5000,6000,7000, 8000, 9000, 10000}; 

        for (auto d : dimensions){
            for(auto p : probes){
                for (auto n : numberOfElementsToCheck){
                    double maxFactor = DBL_MIN;
                    double factor;
                    double sum = 0;

                    double approxSum = 0;
                    double exhaustSum = 0;

                    cube = std::make_shared<HyperCube>(d, p, n, &metric);
                    cube->load_data(images);
                    for(int j = 0; j < retries; j++){
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
                            
                        }
                        sum += maxFactor;
                    }
                    double tA = approxSum / (numberOfQueries*retries);
                    double tE = exhaustSum / (numberOfQueries*retries);
                    printf("Dimensions: %d Probes: %d NumOfElement: %d AverageMaxFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", d, p, n, sum/retries, tA / billion, tE / billion);
                    fflush(stdout);
                    
                }
            }
        }
    }
    // dimensions: 11 probes: 600 numberOfPoints: 4000
    else{
        cube = std::make_shared<HyperCube>(11, 600, 3000, &metric);
        cube->load_data(images);

        std::vector<int> queryNumber = {1000, 2000, 3000, 4000, 5000};
    
        for(auto q : queryNumber){

            double maxFactor = DBL_MIN;
            double factor;
            double approxSum = 0;
            double exhaustSum = 0;

            for(int i = 0; i < q; i++){
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
            }
            double tA = approxSum / q;
            double tE = exhaustSum / q;
            printf("Number of Queries:%d MaxFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", q, maxFactor, tA/billion, tE/billion);
            fflush(stdout);
        }
        
    }
}