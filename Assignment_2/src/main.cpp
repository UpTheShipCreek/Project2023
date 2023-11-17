#include <stdio.h> // Because we like printfs
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "io_functions.h"
#include "mrng.h"

#define QUERY_LIMIT 10

int main(int argc, char **argv){

    bool methodParameter = false;
    bool inputFileParameter = false;
    bool queryFileParameter = false;
    bool outputFileParameter = false;

    int k = 50; 
    int E = 30; 
    int R = 1;
    int N = 1; 
    int l= 20; 
    int m = -1;

    int opt;

    enum methods {gnns = 1, mrng = 2};

    std::string inputFileName, queryFileName, outputFileName;

    FILE* outputFile = fopen("./out/graphSearchResults.out", "w");

    Eucledean metric;

    std::shared_ptr<MonotonicRelativeNeighborGraph> monotonicGraph;
    std::shared_ptr<Graph> genericGraph;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto exhaustTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    while ((opt = getopt(argc, argv, "d:q:k:E:R:N:l:m:o")) != -1){                 
        switch (opt){
            case 'd':                                                               
                inputFileName = optarg;
                inputFileParameter = true;
                break;
            case 'q':
                queryFileName = optarg;
                queryFileParameter = true;
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 'E':                                                             
                E = atoi(optarg);
                break;
            case 'R':
                R = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'l':
                l = atoi(optarg);
                break;
            case 'm':
                m = atoi(optarg);
                methodParameter = true;
                break;
            case 'o':
                outputFileName = optarg;
                outputFileParameter= true;
                break;
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
    std::vector<std::shared_ptr<ImageVector>> queries;

    // Get a correct input file
    do{
        if(inputFileParameter == false){
            printf("Give a path for the input file: ");
            std::cin >> inputFileName;
            dataset = read_mnist_images(inputFileName, 0);
        }
        if(dataset.empty()){
            printf("Error reading input file. Enter another file: \n");
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
    do{
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> approxNearest;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustNearest;

        // Get a correct query file
        do{
            if(queryFileParameter == false){
                printf("Give a path for the query file: ");
                std::cin >> queryFileName;
                queries = read_mnist_images(queryFileName, (int)dataset.size());
            }
            if(queries.empty()){
                printf("Error reading query file. Enter another file: \n");
                queryFileParameter = false;
                continue;
            }
            queryFileParameter = true;
        }while(queries.empty());

        // Get a valid output file if you can but don't dwell on it
        if(outputFileParameter == false){
            printf("Give a path for the output file: ");
            std::cin >> outputFileName;
            outputFile = fopen(outputFileName.c_str(), "w");
            if(outputFile == NULL){
                outputFileName = "./out/graphSearchResults.out";
            }
            outputFile = fopen(outputFileName.c_str(), "w");
            if(outputFile == NULL){
                printf("Error opening output file. Exiting...\n");
                return -1;
            }
            outputFileParameter = true;
        }

        if(m == gnns){
            for(int i = 0; i < QUERY_LIMIT; i++){
                start = std::chrono::high_resolution_clock::now();
                approxNearest = genericGraph->k_nearest_neighbor_search(queries[i], R, GRAPH_DEFAULT_G, E, N);
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;

                start = std::chrono::high_resolution_clock::now();
                exhaustNearest = exhaustive_nearest_neighbor_search_return_images(dataset, queries[i], N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                write_results((int)dataset.size(), queries[i], approxNearest, exhaustNearest, approxTime.count() / 1e9, exhaustTime.count() / 1e9, outputFile);
            }
        }
        else if(m == mrng){
            for(int i = 0; i < QUERY_LIMIT; i++){
                start = std::chrono::high_resolution_clock::now();
                approxNearest = monotonicGraph->k_nearest_neighbor_search(queries[i], l, N);
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;

                start = std::chrono::high_resolution_clock::now();
                exhaustNearest = exhaustive_nearest_neighbor_search_return_images(dataset, queries[i], N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                write_results((int)dataset.size(), queries[i], approxNearest, exhaustNearest, approxTime.count() / 1e9, exhaustTime.count() / 1e9, outputFile);
            }
        }
        // Reset the flag, to show that this query file has been used
        queryFileParameter = false;

        printf("Would you want to try another query file? (y/n): ");
        std::cin >> wantMoreQueries;

    }while(wantMoreQueries == "y");

    return 0;
}