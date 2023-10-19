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
#include "hypercube.h"

#define QUERY_NUMBER 10

long double factorial(int n){
    if(n < 0) return 0;

    int i;
    long double  result = 1; 

    for(i = 1; i <= n; i++){
        result *= i;
    }
    return result;
}

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    int k = 14, M = -1, probes = 2, N = 1;                              //Default values for hypercube projection, if not altered by user
    double R = 1000.0;
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
     while ((opt = getopt(argc, argv, "d:q:k:M:p:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
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
            case 'M':
                M = atoi(optarg);
                break;
            case 'p':
                probes = atoi(optarg);
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
        if(M != -1){
            printf("Program will proceed with values: k = %d, M = %d, probes = %d, N = %d, R = %f\n", k, M, probes, N, R);
        }
        else{
            printf("Program will proceed with values: k = %d, M = Default, probes = %d, N = %d, R = %f\n", k, probes, N, R);
        }
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
    // -------------------------- READ IMAGES ---------------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images(queryFile, 0);
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, (int)queries.size());
    images.insert(images.end(), queries.begin(), queries.end()); // Merge the two vectors of images so you can load them all at once
    // ------------------------------------------------------------------- //
    // -------------------------- READ IMAGES ---------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // -------------- INITIALIZE HYPECUBE AND LOAD IMAGES----------------- //
    // ------------------------------------------------------------------- //
    if(M == -1){// M = (#_of_dataset_images / 2^k) * k!/probes!(k-probes)!  
        M = (int)((images.size()/pow(2, k)) * (factorial(k)/(factorial(probes)*factorial(k-probes))));
    }
    printf("Calculated minimum required value M = %d\n", M);

    HyperCube hypercube(k, probes, M);
    hypercube.load_data(images); // Load the data to the hypercube
    // ------------------------------------------------------------------- //
    // -------------- INITIALIZE HYPECUBE AND LOAD IMAGES----------------- //
    // ------------------------------------------------------------------- //

    
   
    int i = 0;
    while(i < QUERY_NUMBER  && i < (int)(queries.size())){
        // ------------------------------------------------------------------- //
        // ----------------------- APPROXIMATE NEAREST ----------------------- //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now(); // Start the timer
        nearest_approx = hypercube.approximate_k_nearest_neighbors(queries[i], N); // Get the k approximate nearest vectors to the query
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
        range_approx = hypercube.approximate_range_search(queries[i], R); // Get all the images that are in range R from the query
        // ------------------------------------------------------------------- //
        // ------------------------ APPROXIMATE RANGE ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ---------------------------- WRITES ------------------------------- //
        // ------------------------------------------------------------------- //
        write_approx_cube(queries[i], nearest_approx, nearest_exhaust, duration_approx, duration_exhaust, outputFile); 
        write_r_near(range_approx, R, outputFile);
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