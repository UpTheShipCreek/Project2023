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


    // LSH Optimal L: 6 K: 4 Window: 1400 TableSize: 7500
    
    int const retries = 100;
    int const numberOfQueries = 10;
    int const billion = std::pow(10, 9);

    std::shared_ptr<LSH> lsh;

    int mode;
    printf("Press anything other than 1:");
    std::cin >> mode;
    if(mode == 1){
        std::vector<int> lambdas = {5, 6};
        std::vector<int> kappas = {4};
        std::vector<int> windows = {1400}; 
        std::vector<int> tables = {7500, 15000}; 

        // L: 5 K: 4 Window: 1400 TableSize:15000
        // L: 6 K: 4 Window: 1400 TableSize: 7500

        for (auto l : lambdas){
            for(auto k : kappas){
                for (auto w : windows){
                    for(auto t: tables){
                        double maxFactor = DBL_MIN;
                        double factor;
                        double sum = 0;

                        double approxSum = 0;
                        double exhaustSum = 0;

                        lsh = std::make_shared<LSH>(l, k, w, t, &metric);
                        lsh->load_data(images);
                        for(int j = 0; j < retries; j++){
                            for(int i = 0; i < numberOfQueries; i++){
                                int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);


                                start = std::chrono::high_resolution_clock::now();
                                nearestApprox = lsh->approximate_k_nearest_neighbors_return_images(queries[randomIndex], DEFAULT_N);
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
                        printf("L: %d K: %d Window: %d TableSize:%d AverageMaxFactor: %f ApproxAverage: %f ExhaustAverage: %f\n", l, k, w, t, sum/retries, tA / billion, tE / billion);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    // L: 6 K: 4 Window: 1400 TableSize:15000
    else{
        lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric);
        lsh->load_data(images);

        std::vector<int> queryNumber = {1000, 2000, 3000, 4000, 5000, 10000};
    
        for(auto q : queryNumber){

            double maxFactor = DBL_MIN;
            double factor;
            double approxSum = 0;
            double exhaustSum = 0;

            for(int i = 0; i < q; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);


                start = std::chrono::high_resolution_clock::now();
                nearestApprox = lsh->approximate_k_nearest_neighbors_return_images(queries[randomIndex], DEFAULT_N);
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