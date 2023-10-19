#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"
#include "lsh.h"

#define QUERY_NUMBER 10

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    int k = 4, L = 5, N = 1; // Default values  
    double R = 2000.0;                           
    int opt;
    extern char *optarg; 
    std::string inputFile, queryFile, outputFileName;
    int cmdNecessary = 0;
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //


    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
    while ((opt = getopt(argc, argv, "d:q:k:L:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
        switch (opt) {
            case 'd':                                                               //Files
                inputFile = optarg;
                cmdNecessary++;
                break;
            case 'q':
                queryFile = optarg;
                cmdNecessary++;
                break;
            case 'o':
                outputFileName = optarg;
                cmdNecessary++;
                break;
            case 'k':                                                               //Parameters
                k = atoi(optarg);
                break;
            case 'L':
                L = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'R':
                R = atof(optarg);
                break;
        }
    }

    if(cmdNecessary != 3){
        printf("Program execution requires an input file, output file and a query file.");
        return -1;
    }
    else{
        printf("Program will proceed with values: k = %d, L = %d, N = %d, R = %f\n", k, L, N, R);
    }
    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //
    FILE* outputFile = fopen(outputFileName.c_str(), "w");
    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE LSH -------------------------- //
    // ------------------------------------------------------------------- //
    LSH lsh(L, k, MODULO, LSH_TABLE_SIZE);
    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE LSH -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------- METHOD INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::pair<double, int>> nearest_approx;
    std::vector<std::pair<double, int>> nearest_exhaust;
    std::vector<std::pair<double, int>> range_approx;
    // ------------------------------------------------------------------- //
    // ---------------------- METHOD INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_approx = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto duration_exhaust = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ------------------------ OPEN AND LOAD INPUT ---------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images(queryFile, 0);
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, (int)queries.size());
    images.insert(images.end(), queries.begin(), queries.end()); // Merge the two vectors of images so you can load them all at once
    lsh.load_data(images); // Load the data to the LSH
    // ------------------------------------------------------------------- //
    // ------------------------ OPEN AND LOAD INPUT ---------------------- //
    // ------------------------------------------------------------------- //
    int i = 0;
    while(i < QUERY_NUMBER  && i < (int)(queries.size())){
        // ------------------------------------------------------------------- //
        // ----------------------- APPROXIMATE NEAREST ----------------------- //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now(); // Start the timer
        nearest_approx = lsh.approximate_k_nearest_neighbors(queries[i], N); // Get the k approximate nearest vectors to the query
        end  = std::chrono::high_resolution_clock::now(); // End the timer 
        duration_approx = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
        // ------------------------------------------------------------------- //
        // ----------------------- APPROXIMATE NEAREST ----------------------- //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now(); // Start the timer
        nearest_exhaust = exhaustive_nearest_neighbor_search(images, queries[i], N); // Get the k real nearest vectors to the query
        end  = std::chrono::high_resolution_clock::now(); // End the timer 
        duration_exhaust = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
        // ------------------------------------------------------------------- //
        // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ------------------------ APPROXIMATE RANGE ------------------------ //
        // ------------------------------------------------------------------- //
        range_approx = lsh.approximate_range_search(queries[i], R); // Get all the images that are in range R from the query
        // ------------------------------------------------------------------- //
        // ------------------------ APPROXIMATE RANGE ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ---------------------------- WRITES ------------------------------- //
        // ------------------------------------------------------------------- //
        write_approx_lsh(queries[i], nearest_approx, nearest_exhaust, duration_approx, duration_exhaust, outputFile); 
        write_r_near(queries[i], range_approx, outputFile);
        // ------------------------------------------------------------------- //
        // ---------------------------- WRITES ------------------------------- //
        // ------------------------------------------------------------------- //
        i++;
    }
    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //
    fclose(outputFile);
    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //

    return 0;
}