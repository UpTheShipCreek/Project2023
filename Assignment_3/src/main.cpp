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
    std::vector<std::pair<double, int>> nearestApproxNumber;
    std::vector<std::pair<double, int>> nearestExhaustNumber;
    std::vector<double> nearestApproxInitial;
    std::vector<double> nearestExhaustInitial;
    std::vector<double> queryInitial;


    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto exhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    Random rand;

    // Images
    // std::string datasetFilename = "./in/input.dat";
    // std::string querysetFilename = "./in/query.dat";
    // std::string datasetFilename = "./in/encoded_dataset.dat";
    // std::string querysetFilename = "./in/encoded_queryset.dat";


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

    SpaceCorrespondace datasetSpaceCorrespondace(dataset, reducedDataset);
    SpaceCorrespondace querysetSpaceCorrespondace(queryset, reducedQueryset);

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

    int dimensions = reducedDatasetHeaderInfo->get_numberOfRows() * reducedDatasetHeaderInfo->get_numberOfColumns();

    int const retries = 100;
    int const numberOfQueries = 10;
    int const billion = std::pow(10, 9);

    std::shared_ptr<LSH> lsh;

    int mode;

    printf("Press anything other than 1:");
    std::cin >> mode;
    if(mode == 1){
        std::vector<int> lambdas = {3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> kappas = {3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> windows = {70, 100, 130, 160, 190}; 
        std::vector<int> tables = {3750, 7500}; 
        for (auto l : lambdas){
            for(auto k : kappas){
                for (auto w : windows){
                    for(auto t: tables){
                        double maxFactor = DBL_MIN;
                        double factor;
                        double sum = 0;

                        double approxSum = 0;
                        double exhaustSum = 0;

                        lsh = std::make_shared<LSH>(l, k, w, t, &metric, dimensions);
                        lsh->load_data(reducedDataset);
                        for(int j = 0; j < retries; j++){
                            for(int i = 0; i < numberOfQueries; i++){
                                int randomIndex = rand.generate_int_uniform(0, (int)reducedQueryset.size() - 1);


                                start = std::chrono::high_resolution_clock::now();
                                nearestApproxNumber = lsh->approximate_k_nearest_neighbors(reducedQueryset[randomIndex], DEFAULT_N);
                                if(nearestApproxNumber.size() == 0){
                                    printf("Error: No nearest neighbors found\n");
                                    break;
                                }
                                end = std::chrono::high_resolution_clock::now();
                                end = std::chrono::high_resolution_clock::now();
                                approxTime = end - start;
                                
                                start = std::chrono::high_resolution_clock::now();
                                nearestExhaustNumber = exhaustive_nearest_neighbor_search(reducedDataset, reducedQueryset[randomIndex], DEFAULT_N, &metric);
                                end = std::chrono::high_resolution_clock::now();
                                exhaustTime = end - start;

                                nearestApproxInitial = datasetSpaceCorrespondace.get_initial(nearestApproxNumber[0].second);
                                nearestExhaustInitial = datasetSpaceCorrespondace.get_initial(nearestExhaustNumber[0].second);
                                queryInitial = querysetSpaceCorrespondace.get_initial(reducedQueryset[randomIndex]->get_number());

                                factor = metric.calculate_distance(nearestApproxInitial, queryInitial) / metric.calculate_distance(nearestExhaustInitial, queryInitial);
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
        lsh = std::make_shared<LSH>(3, 3, 100, 3750, &metric, dimensions);
        lsh->load_data(reducedDataset);

        std::vector<int> queryNumber = {1000, 2000, 3000, 4000, 5000, 10000};
    
        for(auto q : queryNumber){

            double maxFactor = DBL_MIN;
            double factor;
            double approxSum = 0;
            double exhaustSum = 0;

            for(int i = 0; i < q; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)reducedQueryset.size() - 1);


                start = std::chrono::high_resolution_clock::now();
                nearestApproxNumber = lsh->approximate_k_nearest_neighbors(reducedQueryset[randomIndex], DEFAULT_N);
                if(nearestApproxNumber.size() == 0){
                    printf("Error: No nearest neighbors found\n");
                    break;
                }
                end = std::chrono::high_resolution_clock::now();
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;
                
                start = std::chrono::high_resolution_clock::now();
                nearestExhaustNumber = exhaustive_nearest_neighbor_search(reducedDataset, reducedQueryset[randomIndex], DEFAULT_N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                nearestApproxInitial = datasetSpaceCorrespondace.get_initial(nearestApproxNumber[0].second);
                nearestExhaustInitial = datasetSpaceCorrespondace.get_initial(nearestExhaustNumber[0].second);

                factor = metric.calculate_distance(nearestApproxInitial, queryInitial) / metric.calculate_distance(nearestExhaustInitial, queryInitial);
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