#include <stdio.h>
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "io_functions.h"
#include "mrng.h"

#define DEFAULT_N 10
#define QUERY_LIMIT 10
#define GREEDY_STEPS 10

int main(void){

    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestExhaustReduced;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> optimalNearestApprox;
    std::shared_ptr<ImageVector> nearestExhaustReducedCorrespondant;
    std::vector<double> nearestExhaustInitial;
    std::vector<double> queryInitial;


    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto exhaustReducedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto optimalApproxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    Random rand;

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images("./in/input.dat", 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedDatasetInfo = read_mnist_images("./in/encoded_dataset.dat", 0);
    HeaderInfo* reducedDatasetHeaderInfo = reducedDatasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedDataset = reducedDatasetInfo.second;

    
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> querysetInfo = read_mnist_images("./in/query.dat", (int)dataset.size());
    HeaderInfo* querysetHeaderInfo = querysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> queryset = querysetInfo.second;

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedQuerysetInfo = read_mnist_images("./in/encoded_queryset.dat", (int)dataset.size());
    HeaderInfo* reducedQuerysetHeaderInfo = reducedQuerysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedQueryset = reducedQuerysetInfo.second;

    SpaceCorrespondace datasetSpaceCorrespondace(dataset);
    SpaceCorrespondace querysetSpaceCorrespondace(queryset);

    if(!(*datasetHeaderInfo == *querysetHeaderInfo)){
        printf("Dataset and queryset header info do not match\n");
        return -1;
    }

    // Check the numbers
    printf(
        "Dataset size: %d vs %d\nQueryset size: %d vs %d\nNumber of Bytes: %d\n", 
        (int)reducedDataset.size(), reducedDatasetHeaderInfo->get_numberOfImages(),
        (int)reducedQueryset.size(), reducedQuerysetHeaderInfo->get_numberOfImages(),
        reducedDatasetHeaderInfo->get_numberOfRows() * reducedDatasetHeaderInfo->get_numberOfColumns()
    );

    int const retries = 100;
    int const numberOfQueries = 10;
    int const billion = std::pow(10, 9);

    LSH quick_nn(6, 4, 1400, 7500, &metric, 784); 
    quick_nn.load_data(dataset);

    int foundCount = 0;
    
    // int mode;
    // printf("To run the method with different parameters press 1:");
    // std::cin >> mode;
    if(1 == 0){
        double maxFactor = DBL_MIN;
        double factor;
        double sum = 0;

        double exhaustReducedSum = 0;
        double optimalApproxSum = 0;
        
        for(int j = 0; j < retries; j++){
            for(int i = 0; i < numberOfQueries; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)reducedQueryset.size() - 1);

                start = std::chrono::high_resolution_clock::now();
                nearestExhaustReduced = exhaustive_nearest_neighbor_search_return_images(reducedDataset,reducedQueryset[randomIndex], DEFAULT_N, &metric);
                if(nearestExhaustReduced.size() == 0){
                    printf("Error: No nearest neighbors found\n");
                    break;
                }
                end = std::chrono::high_resolution_clock::now();
                end = std::chrono::high_resolution_clock::now();
                exhaustReducedTime = end - start;
                
                start = std::chrono::high_resolution_clock::now();
                optimalNearestApprox = quick_nn.approximate_k_nearest_neighbors_return_images(queryset[randomIndex], DEFAULT_N);
                end = std::chrono::high_resolution_clock::now();
                optimalApproxTime = end - start;

                nearestExhaustReducedCorrespondant = datasetSpaceCorrespondace.get_initial(nearestExhaustReduced[0].second->get_number());
                
                if(nearestExhaustReduced[0].second->get_number() == optimalNearestApprox[0].second->get_number()){
                    foundCount++;
                }
                
                factor = metric.calculate_distance(nearestExhaustReducedCorrespondant->get_coordinates(), queryset[randomIndex]->get_coordinates()) / optimalNearestApprox[0].first;
                if(factor > maxFactor){
                    maxFactor = factor;
                }
                exhaustReducedSum += exhaustReducedTime.count();
                optimalApproxSum += optimalApproxTime.count();
            }
            sum += maxFactor;
        }
        double tRE = exhaustReducedSum / (numberOfQueries*retries);
        double tOA = optimalApproxSum / (numberOfQueries*retries);
        printf("AverageMaxFactor: %f Reduced Exhuast: %f Optimal Approx: %f Found Optimal: %d\n", sum/retries, tRE / billion, tOA / billion, foundCount);
        fflush(stdout);
    }
    else{
        std::vector<int> queryNumber = {1000, 2000, 3000, 4000, 5000};
    
        for(auto q : queryNumber){

            double maxFactor = DBL_MIN;
            double factor;
            double exhaustReducedSum = 0;
            double optimalApproxSum = 0;

            for(int i = 0; i < q; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)reducedQueryset.size() - 1);

                start = std::chrono::high_resolution_clock::now();
                nearestExhaustReduced = exhaustive_nearest_neighbor_search_return_images(reducedDataset,reducedQueryset[randomIndex], DEFAULT_N, &metric);
                if(nearestExhaustReduced.size() == 0){
                    printf("Error: No nearest neighbors found\n");
                    break;
                }
                end = std::chrono::high_resolution_clock::now();
                end = std::chrono::high_resolution_clock::now();
                exhaustReducedTime = end - start;
                
                start = std::chrono::high_resolution_clock::now();
                optimalNearestApprox = quick_nn.approximate_k_nearest_neighbors_return_images(queryset[randomIndex], DEFAULT_N);
                end = std::chrono::high_resolution_clock::now();
                optimalApproxTime = end - start;

                nearestExhaustReducedCorrespondant = datasetSpaceCorrespondace.get_initial(nearestExhaustReduced[0].second->get_number());

                // printf("Approx: %d Exhaust: %d Query: %d\n", nearestExhaustReduced[0].second->get_number(), optimalNearestApprox[0].second, reducedQueryset[randomIndex]->get_number());

                factor = metric.calculate_distance(nearestExhaustReducedCorrespondant->get_coordinates(), queryset[randomIndex]->get_coordinates()) / optimalNearestApprox[0].first;
                if(factor > maxFactor){
                    maxFactor = factor;
                }
                exhaustReducedSum += exhaustReducedTime.count();
                optimalApproxSum += optimalApproxTime.count();
            }
            double tRE = exhaustReducedSum / q;
            double tOA = optimalApproxSum / q;
            printf("Number of Queries:%d MaxFactor: %f Reduced Exhuast: %f Optimal Approx: %f\n", q, maxFactor, tRE/billion, tOA/billion);
            fflush(stdout);
        }
    }
}