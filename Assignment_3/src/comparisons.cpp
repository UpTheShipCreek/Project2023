#include <stdio.h>
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "io_functions.h"
#include "mrng.h"

#define DEFAULT_N 10
#define MRNG_L_FACTOR 0.001
#define HYPERCUBE_M_FACTOR 0.06
#define HYPERCUBE_PROBES_FACTOR 0.01

double calculate_average_approximation_factor(std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestNeighbours, std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestNeighboursApprox){
    double sum = 0;
    for(int i = 0; i < (int)nearestNeighboursApprox.size(); i++){
        sum += nearestNeighboursApprox[i].first / nearestNeighbours[i].first;
    }
    return sum / (double)nearestNeighbours.size();
}


int main(int argc, char **argv){
    int const billion = std::pow(10, 9);

    // Metric
    Eucledean metric;
    
    // Random
    Random rand;

    // Output file
    FILE* outputFile = fopen("./out/comparisons_details.out", "w");

    // Clock
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    // Brute force time
    auto trueExhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Original space method times
    auto lshTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto hypercubeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto gnnsTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto mrngTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Reduced space method times
    auto reducedExhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedGnnsTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto reducedMrngTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);


    std::string inputFileName, queriesFileName, reducedInputFileName, reducedQueriesFileName;

    if(argc != 6){
        printf("Error: Argument Number. Example call: ./comparisons <original dataset> <original queryset> <reduced dataset> <reduced queryset> <number of queries>\n");
        return -1;
    }
    inputFileName = argv[1];
    queriesFileName = argv[2];
    reducedInputFileName = argv[3];
    reducedQueriesFileName = argv[4];
    int numberOfQueries = atoi(argv[5]);

    // Read the original sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> datasetInfo = read_mnist_images(inputFileName, 0);
    HeaderInfo* datasetHeaderInfo = datasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> dataset = datasetInfo.second;
    if(dataset.empty()){
        printf("Error reading file: %s\n", inputFileName.c_str());
        return -1;
    }
    if((int)dataset.size() != datasetHeaderInfo->get_numberOfImages()){
        printf("Warning: Dataset size does not match the header info (%d vs %d)\n", (int)dataset.size(), datasetHeaderInfo->get_numberOfImages());
    }

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> querysetInfo = read_mnist_images(queriesFileName, (int)dataset.size());
    HeaderInfo* querysetHeaderInfo = querysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> queryset = querysetInfo.second;
    if(queryset.empty()){
        printf("Error reading file: %s\n", queriesFileName.c_str());
        return -1;
    }
    if((int)queryset.size() != querysetHeaderInfo->get_numberOfImages()){
        printf("Warning: Queryset size does not match the header info (%d vs %d)\n", (int)queryset.size(), querysetHeaderInfo->get_numberOfImages());
    }

    // Check that the shapes match between the two original sets
    if(!(*datasetHeaderInfo == *querysetHeaderInfo)){
        printf("Dataset and queryset shapes do not match\n");
        return -1;
    }

    int originalDimensions = datasetHeaderInfo->get_numberOfRows() * datasetHeaderInfo->get_numberOfColumns();

    // Read the reduced sets from the file 
    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedDatasetInfo = read_mnist_images(reducedInputFileName, 0);
    HeaderInfo* reducedDatasetHeaderInfo = reducedDatasetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedDataset = reducedDatasetInfo.second;
    if(reducedDataset.empty()){
        printf("Error reading file: %s\n", reducedInputFileName.c_str());
        return -1;
    }
    if((int)reducedDataset.size() != reducedDatasetHeaderInfo->get_numberOfImages()){
        printf("Warning: Reduced dataset size does not match the header info (%d vs %d)\n", (int)reducedDataset.size(), reducedDatasetHeaderInfo->get_numberOfImages());
    }

    std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> reducedQuerysetInfo = read_mnist_images(reducedQueriesFileName, (int)reducedDataset.size());
    HeaderInfo* reducedQuerysetHeaderInfo = reducedQuerysetInfo.first.get();
    std::vector<std::shared_ptr<ImageVector>> reducedQueryset = reducedQuerysetInfo.second;
    if(reducedQueryset.empty()){
        printf("Error reading file: %s\n", reducedQueriesFileName.c_str());
        return -1;
    }
    if((int)reducedQueryset.size() != reducedQuerysetHeaderInfo->get_numberOfImages()){
        printf("Warning: Reduced queryset size does not match the header info (%d vs %d)\n", (int)reducedQueryset.size(), reducedQuerysetHeaderInfo->get_numberOfImages());
    }

    // Check that the shapes match between the two reduced sets
    if(!(*reducedDatasetHeaderInfo == *reducedQuerysetHeaderInfo)){
        printf("Reduced dataset and queryset shapes do not match\n");
        return -1;
    }
    int reducedDimensions = reducedDatasetHeaderInfo->get_numberOfRows() * reducedDatasetHeaderInfo->get_numberOfColumns();
    
    // Set up the correspondances, we can then use the number of the image to get the original coordinates with get_initial
    SpaceCorrespondace datasetSpaceCorrespondace(dataset);

    // Set up the methods for the Original Space
    // LSH
    std::shared_ptr<LSH> lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric, originalDimensions); 
    lsh->load_data(dataset);

    // Hypercube
    int probes = (int)(HYPERCUBE_PROBES_FACTOR * (double)dataset.size());
    int M = (int)(HYPERCUBE_M_FACTOR * (double)dataset.size());
    std::shared_ptr<HyperCube> hypercube = std::make_shared<HyperCube>(11, probes, M, WINDOW, &metric, originalDimensions);
    hypercube->load_data(dataset);

    // GNNS
    std::shared_ptr<Graph> gnns = std::make_shared<Graph>(dataset, &metric);

    start = std::chrono::high_resolution_clock::now();
    gnns->initialize_neighbours_approximate_method(lsh, 50);
    end = std::chrono::high_resolution_clock::now();
    auto gnnsInitializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double gnnsIndexCreationTime = gnnsInitializationTime.count() / 1e9;
    printf("Original GNNS initialization time: %f\n", gnnsIndexCreationTime);
    
    // MRNG
    int l = (int)(MRNG_L_FACTOR * (double)dataset.size());
    start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<MonotonicRelativeNeighborGraph> mrng = std::make_shared<MonotonicRelativeNeighborGraph>(dataset, lsh, l, &metric);
    end = std::chrono::high_resolution_clock::now();
    auto mrngInitializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double mrngIndexCreationTime = mrngInitializationTime.count() / 1e9;
    printf("Original MRNG initialization time: %f\n", mrngIndexCreationTime);
    

    // Set up the methods for the Reduced Space
    // Reduced LSH
    printf("Reduced dimensions: %d\n",reducedDimensions);
    std::shared_ptr<LSH> reducedLsh = std::make_shared<LSH>(8, 2, 9, 7500, &metric, reducedDimensions);
    reducedLsh->load_data(reducedDataset);

    // GNNS
    std::shared_ptr<Graph> reducedGnns = std::make_shared<Graph>(reducedDataset, &metric);
    start = std::chrono::high_resolution_clock::now();
    reducedGnns->initialize_neighbours_approximate_method(reducedLsh, 50);
    end = std::chrono::high_resolution_clock::now();
    auto reducedGnnsInitializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double reducedGnnsIndexCreationTime = reducedGnnsInitializationTime.count() / 1e9;
    printf("Reduced GNNS initialization time: %f\n", reducedGnnsIndexCreationTime);
    
    // MRNG
    start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<MonotonicRelativeNeighborGraph> reducedMrng = std::make_shared<MonotonicRelativeNeighborGraph>(reducedDataset, reducedLsh, l, &metric);
    end = std::chrono::high_resolution_clock::now();
    auto reducedMrngInitializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double reducedMrngIndexCreationTime = reducedMrngInitializationTime.count() / 1e9;
    printf("Reduced MRNG initialization time: %f\n", reducedMrngIndexCreationTime);
    fflush(stdout);

    // Search 
    std::vector<int> queriesInRowNumbers = {numberOfQueries};

    printf("Collecting results...\n");

    for(auto& queriesInRow : queriesInRowNumbers){

        // Sum of times
        double trueExhaustTimeSum = 0;
        double lshTimeSum = 0;
        double hypercubeTimeSum = 0;
        double gnnsTimeSum = 0;
        double mrngTimeSum = 0;
        double reducedExhaustTimeSum = 0;
        double reducedGnnsTimeSum = 0;
        double reducedMrngTimeSum = 0;

        // Approximation factors
        double lshAAF = 0;
        double hypercubeAAF = 0;
        double gnnsAAF = 0;
        double mrngAAF = 0;
        double reducedExhaustAAF = 0;
        double reducedGnnsAAF= 0;
        double reducedMrngAAF = 0;



        for(int i = 0; i < queriesInRow; i++){
            int randomIndex = rand.generate_int_uniform(0, (int)queryset.size() - 1);

            // Original Space
            // True
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestTrue = exhaustive_nearest_neighbor_search_return_images(dataset, queryset[randomIndex], DEFAULT_N, &metric);
            end = std::chrono::high_resolution_clock::now();
            trueExhaustTime = end - start;
            trueExhaustTimeSum += trueExhaustTime.count();

            // LSH
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestLSH = lsh->approximate_k_nearest_neighbors_return_images(queryset[randomIndex], DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            if(nearestLSH.empty()){
                printf("Failed approximation: LSH\n");
                fflush(stdout);
            }
            else{
                lshTime = end - start;
                lshTimeSum += lshTime.count();
                lshAAF += calculate_average_approximation_factor(nearestTrue, nearestLSH);
            }

            fprintf(outputFile, "Original LSH: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestLSH, nearestTrue, outputFile);

            // Hypercube
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestHypercube = hypercube->approximate_k_nearest_neighbors_return_images(queryset[randomIndex], DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            if(nearestHypercube.empty()){
                printf("Failed approximation: Hypercube\n");
                fflush(stdout);
            }
            else{
                hypercubeTime = end - start;
                hypercubeTimeSum += hypercubeTime.count();
                hypercubeAAF += calculate_average_approximation_factor(nearestTrue, nearestHypercube);
            }
            fprintf(outputFile, "Original Hypercube: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestHypercube, nearestTrue, outputFile);

            // GNNS
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestGnns = gnns->k_nearest_neighbor_search(queryset[randomIndex], 3, 10, 20, DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            if(nearestGnns.empty()){
                printf("Failed approximation: GNNS\n");
                fflush(stdout);
            }
            else{
                gnnsTime = end - start;
                gnnsTimeSum += gnnsTime.count();
                gnnsAAF += calculate_average_approximation_factor(nearestTrue, nearestGnns);
            }
            fprintf(outputFile, "Original GNNS: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestGnns, nearestTrue, outputFile);

            // MRNG
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestMrng = mrng->k_nearest_neighbor_search(queryset[randomIndex], l, DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            if(nearestMrng.empty()){
                printf("Failed approximation: MRNG\n");
                fflush(stdout);
            }
            else{
                mrngTime = end - start;
                mrngTimeSum += mrngTime.count();
                mrngAAF += calculate_average_approximation_factor(nearestTrue, nearestMrng);
            }
            fprintf(outputFile, "Original MRNG: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestMrng, nearestTrue, outputFile);


            // Reduced Space
            // Exhaustive
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedExhaust = exhaustive_nearest_neighbor_search_return_images(reducedDataset, reducedQueryset[randomIndex], DEFAULT_N, &metric);
            end = std::chrono::high_resolution_clock::now();
        
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedExhaustDistanceCorrespondace;
            for(auto& image : nearestReducedExhaust){
                nearestReducedExhaustDistanceCorrespondace.push_back({
                    metric.calculate_distance(datasetSpaceCorrespondace.get_initial(image.second->get_number())->get_coordinates(), queryset[randomIndex]->get_coordinates())
                    ,
                    image.second
                });
            }
            reducedExhaustTime = end - start;
            reducedExhaustTimeSum += reducedExhaustTime.count();
            reducedExhaustAAF += calculate_average_approximation_factor(nearestTrue, nearestReducedExhaustDistanceCorrespondace);
            fprintf(outputFile, "Reduced Exhaustive: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestReducedExhaustDistanceCorrespondace, nearestTrue, outputFile);

            // Reduced GNNS
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedGnns = reducedGnns->k_nearest_neighbor_search(reducedQueryset[randomIndex], 3, 10, 20, DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedGnnsDistanceCorrespondace;
            
            if(nearestReducedGnns.empty()){
                printf("Failed approximation: Reduced GNNS\n");
            }
            else{
                for(auto& image : nearestReducedGnns){
                    nearestReducedGnnsDistanceCorrespondace.push_back({
                        metric.calculate_distance(datasetSpaceCorrespondace.get_initial(image.second->get_number())->get_coordinates(), queryset[randomIndex]->get_coordinates())
                        ,
                        image.second
                    });
                }
                reducedGnnsTime = end - start;
                reducedGnnsTimeSum += reducedGnnsTime.count();
                reducedGnnsAAF += calculate_average_approximation_factor(nearestTrue, nearestReducedGnnsDistanceCorrespondace);
            }
            fprintf(outputFile, "Reduced GNNS: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestReducedGnnsDistanceCorrespondace, nearestTrue, outputFile);

            // MRNG
            start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedMrng = reducedMrng->k_nearest_neighbor_search(reducedQueryset[randomIndex], l, DEFAULT_N);
            end = std::chrono::high_resolution_clock::now();
            
            std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestReducedMrngDistanceCorrespondace;
           
            if(nearestReducedMrng.empty()){
                printf("Failed approximation: Reduced MRNG\n");
            }
            else{  
                for(auto& image : nearestReducedMrng){
                    nearestReducedMrngDistanceCorrespondace.push_back({
                        metric.calculate_distance(datasetSpaceCorrespondace.get_initial(image.second->get_number())->get_coordinates(), queryset[randomIndex]->get_coordinates())
                        ,
                        image.second
                    });
                }
                reducedMrngTime = end - start;
                reducedMrngTimeSum += reducedMrngTime.count();
                reducedMrngAAF += calculate_average_approximation_factor(nearestTrue, nearestReducedMrngDistanceCorrespondace);
            }
            fprintf(outputFile, "Reduced MRNG: \n");
            write_results((int)dataset.size(), queryset[randomIndex], nearestReducedMrngDistanceCorrespondace, nearestTrue, outputFile);
        }
        // Calculate the average times
        double averageTrueExhaustTime = trueExhaustTimeSum / (double)queriesInRow;
        double averageLshTime = lshTimeSum/ (double)queriesInRow;
        double averageHypercubeTime = hypercubeTimeSum / (double)queriesInRow;
        double averageGnnsTime = gnnsTimeSum / (double)queriesInRow;
        double averageMrngTime = mrngTimeSum / (double)queriesInRow;
        double averageReducedExhaustTime = reducedExhaustTimeSum / (double)queriesInRow;
        double averageReducedGnnsTime = reducedGnnsTimeSum / (double)queriesInRow;
        double averageReducedMrngTime = reducedMrngTimeSum / (double)queriesInRow;

        // Calculate the average AAF
        double averageLshAAF = lshAAF / (double)queriesInRow;
        double averageHypercubeAAF = hypercubeAAF / (double)queriesInRow;
        double averageGnnsAAF = gnnsAAF / (double)queriesInRow;
        double averageMrngAAF = mrngAAF / (double)queriesInRow;
        double averageReducedExhaustAAF = reducedExhaustAAF / (double)queriesInRow;
        double averageReducedGnnsAAF = reducedGnnsAAF / (double)queriesInRow;
        double averageReducedMrngAAF = reducedMrngAAF / (double)queriesInRow;

        // Print the average times and AAF for each method
        printf("Queries in row: %d\n", queriesInRow);
        printf("True Exhaustive: %f\n", averageTrueExhaustTime / billion);
        printf("LSH: %f AAF: %f\n", averageLshTime / billion, averageLshAAF);
        printf("Hypercube: %f AAF: %f\n", averageHypercubeTime / billion, averageHypercubeAAF);
        printf("GNNS: %f AAF: %f\n", averageGnnsTime / billion, averageGnnsAAF);
        printf("MRNG: %f AAF: %f\n", averageMrngTime / billion, averageMrngAAF);
        printf("Reduced Exhaustive: %f AAF: %f\n", averageReducedExhaustTime / billion, averageReducedExhaustAAF);
        printf("Reduced GNNS: %f AAF: %f\n", averageReducedGnnsTime / billion, averageReducedGnnsAAF);
        printf("Reduced MRNG: %f AAF: %f\n", averageReducedMrngTime / billion, averageReducedMrngAAF);
        printf("\n");
    }
    fclose(outputFile);
    return 0;
}