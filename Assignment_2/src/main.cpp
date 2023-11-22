#include <stdio.h> // Because we like printfs
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "io_functions.h"
#include "mrng.h"

#define QUERY_LIMIT 10
#define GREEDY_STEPS 10

int main(int argc, char **argv){

    bool outputFileIsOpen = false;

    bool methodParameter = false;
    bool inputFileParameter = false;
    bool queryFileParameter = false;
    bool outputFileParameter = false;

    int k = 50; 
    int E = 30; 
    int R = 1;
    int N = 1; 
    int l = 20; 
    int m = -1;

    enum methods {gnns = 1, mrng = 2};

    std::string inputFileName, queryFileName, outputFileName;

    FILE* outputFile = NULL;

    Eucledean metric;

    std::shared_ptr<MonotonicRelativeNeighborGraph> monotonicGraph;
    std::shared_ptr<Graph> genericGraph;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto exhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    for (int i = 1; i < argc; i++){ // Start from 1 to skip the program name (argv[0])
        std::string arg = argv[i];
        if(arg == "-d"){
            if(i + 1 < argc){
                inputFileName = argv[i + 1];
                inputFileParameter = true;
                i++; // Skip the next argument
            }
        } 
        else if(arg == "-q"){
            if(i + 1 < argc){
                queryFileName = argv[i + 1];
                queryFileParameter = true;
                i++;
            }
        } 
        else if(arg == "-k"){
            if(i + 1 < argc){
                k = atoi(argv[i + 1]);
                i++;
            }
        }
        else if(arg == "-E"){
            if(i + 1 < argc){
                E = atoi(argv[i + 1]);
                i++;
            }
        }
        else if(arg == "-R"){
            if(i + 1 < argc){
                R = atoi(argv[i + 1]);
                i++;
            }
        }
        else if(arg == "-N"){
            if(i + 1 < argc){
                N = atoi(argv[i + 1]);
                i++;
            }
        }
        else if(arg == "-l"){
            if(i + 1 < argc){
                l = atoi(argv[i + 1]);
                i++;
            }
        }
        else if(arg == "-m"){
            if(i + 1 < argc){
                m = atoi(argv[i + 1]);
                methodParameter = true;
                i++;
            }
        } 
        else if(arg == "-o"){
            if(i + 1 < argc){
                outputFileName = argv[i + 1];
                outputFileParameter = true;
                i++;
            }
        } 
    }

    if(methodParameter == false){
        printf("Graph search method parameter missing.\n");
        return -1;
    }
    if(m != 1 && m != 2){
        printf("Invalid graph search method parameter.\n");
        return -1;
    }

    std::string wantMoreQueries;
    std::vector<std::shared_ptr<ImageVector>> dataset;

    // Get a correct input file
    do{
        if(inputFileParameter == false){
            printf("Give a path for the input file: ");
            std::cin >> inputFileName;
            dataset = read_mnist_images(inputFileName, 0);
        }
        else{
            dataset = read_mnist_images(inputFileName, 0);
        }
        if(dataset.empty()){
            printf("Error reading input file.\n");
            inputFileParameter = false;
            continue;
        }
        inputFileParameter = true;
    }while(dataset.empty());


    if(m == gnns){
        std::shared_ptr<LSH> lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric);
        genericGraph = std::make_shared<Graph>(dataset, &metric);
        genericGraph->initialize_neighbours_approximate_method(lsh, k);
    }
    else if(m == mrng){
        monotonicGraph = std::make_shared<MonotonicRelativeNeighborGraph>(dataset, &metric);
    }
    else{
        printf("Graph search method parameter missing. Exiting... \n");
        return -1;
    }

    // Program loop
    int imagesAlreadyRead = (int)dataset.size();
    do{
        std::vector<std::shared_ptr<ImageVector>> queries;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> approxNearest;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustNearest;

        // Get a correct query file
        do{
            if(queryFileParameter == false){
                printf("Give a path for the query file: ");
                std::cin >> queryFileName;
                queries = read_mnist_images(queryFileName, imagesAlreadyRead);
            }
            else{
                queries = read_mnist_images(queryFileName, imagesAlreadyRead);
            }
            if(queries.empty()){
                printf("Error reading query file.\n");
                queryFileParameter = false;
                continue;
            }
            queryFileParameter = true;
            imagesAlreadyRead += (int)queries.size();
        }while(queries.empty());

        // Get a valid output file if you can but don't dwell on it
        if(outputFileParameter == false){
            printf("Give a path for the output file: ");
            std::cin >> outputFileName;
            outputFile = fopen(outputFileName.c_str(), "w");
            if(outputFile == NULL){
                outputFileName = "./out/graphSearchResults.out";
                outputFile = fopen(outputFileName.c_str(), "w");
                if(outputFile == NULL){
                    perror("Error opening output file. Exiting...\n");
                    return -1;
                }
            }
            outputFileParameter = true;
            outputFileIsOpen = true;
        }
        else{
            if(outputFileIsOpen == false){
                outputFile = fopen(outputFileName.c_str(), "w");
                if(outputFile == NULL){
                    outputFileName = "./out/graphSearchResults.out";
                    outputFile = fopen(outputFileName.c_str(), "w");
                    if(outputFile == NULL){
                        perror("Error opening output file. Exiting...\n");
                        return -1;
                    }
                }
            }
            outputFileIsOpen = true;
        }

        if(m == gnns){
            int numOfQueries;
            double MAF = DBL_MIN;
            double averageApprox = 0;
            double averageTrue = 0;

            if(QUERY_LIMIT <= (int)queries.size()){
                numOfQueries = QUERY_LIMIT;
            }
            else{
                numOfQueries = (int)queries.size();
            }

            for(int i = 0; i < numOfQueries; i++){
                start = std::chrono::high_resolution_clock::now();
                approxNearest = genericGraph->k_nearest_neighbor_search(queries[i], R, GREEDY_STEPS, E, N);
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;

                start = std::chrono::high_resolution_clock::now();
                exhaustNearest = exhaustive_nearest_neighbor_search_return_images(dataset, queries[i], N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                if(approxNearest[0].first / exhaustNearest[0].first > MAF){
                    MAF = approxNearest[0].first / exhaustNearest[0].first;
                }

                averageApprox += approxTime.count() / 1e9;
                averageTrue += exhaustTime.count() / 1e9;

                write_results((int)dataset.size(), queries[i], approxNearest, exhaustNearest, outputFile);
            }
            fprintf(outputFile, "tAverageApproximate: %f\ntAverageTrue: %f\nMAF: %f\n\n", averageApprox/numOfQueries, averageTrue/numOfQueries, MAF);
            fflush(outputFile);
        }
        else if(m == mrng){
            int numOfQueries;
            double MAF = DBL_MIN;
            double averageApprox = 0;
            double averageTrue = 0;

            if(QUERY_LIMIT <= (int)queries.size()){
                numOfQueries = QUERY_LIMIT;
            }
            else{
                numOfQueries = (int)queries.size();
            }

            for(int i = 0; i < numOfQueries; i++){
                start = std::chrono::high_resolution_clock::now();
                approxNearest = monotonicGraph->k_nearest_neighbor_search(queries[i], l, N);
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;

                start = std::chrono::high_resolution_clock::now();
                exhaustNearest = exhaustive_nearest_neighbor_search_return_images(dataset, queries[i], N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                if(approxNearest[0].first / exhaustNearest[0].first > MAF){
                    MAF = approxNearest[0].first / exhaustNearest[0].first;
                }

                averageApprox += approxTime.count() / 1e9;
                averageTrue += exhaustTime.count() / 1e9;

                write_results((int)dataset.size(), queries[i], approxNearest, exhaustNearest, outputFile);
            }
            fprintf(outputFile, "tAverageApproximate: %f\ntAverageTrue: %f\nMAF: %f\n\n", averageApprox/numOfQueries, averageTrue/numOfQueries, MAF);
            fflush(outputFile);
        }
        // Reset the flag, to show that this query file has been used
        queryFileParameter = false;

        printf("Would you want to try another query file? (y/n): ");
        std::cin >> wantMoreQueries;

    }while(wantMoreQueries == "y");

    if(outputFileIsOpen){
        fclose(outputFile);
    }
    return 0;
}
