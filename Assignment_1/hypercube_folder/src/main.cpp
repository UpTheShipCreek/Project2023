#include <iostream>
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

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    int k = 14, M = 2000, probes = 500, N = 1; // Defaults                 
    double R = 10000.0;
    int opt;
    extern char *optarg; 
    std::string inputFile, queryFile; 
    std::string outputFileName = "./out/cube.out";
    FILE* outputFile;
    int cmdNecessary = 0;
    Eucledean metric;
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------- METHOD INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> range_approx;
    std::vector<std::pair<double, int>> nearest_approx_num;
    std::vector<std::pair<double, int>> range_approx_num;
    std::vector<std::pair<double, int>> nearest_exhaust;
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
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
    if(argc > 1){
        while ((opt = getopt(argc, argv, "d:q:k:M:p:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
            switch (opt){
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
            printf("Program execution requires an input file, output file and a query file.\nInput example: ./cube –d <input file> –q <query file> –k <int> -M <int> -probes <int> -ο <output file> -Ν <number of nearest> -R <radius>\n");
            return -1;
        }
        else{
            printf("Program will proceed with values: k = %d, M = %d, probes = %d, N = %d, R = %f\n", k, M, probes, N, R);
        }
        // ------------------------------------------------------------------- //
        // -------------------------- INPUT PARSING -------------------------- //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // -------------------------- OPEN OUTPUT FILE ----------------------- //
        // ------------------------------------------------------------------- //
        outputFile = fopen(outputFileName.c_str(), "w");
        if(outputFile == NULL){
            perror("Error opening output file");
            return -1;
        }
        // ------------------------------------------------------------------- //
        // -------------------------- OPEN OUTPUT FILE ----------------------- //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ---------------------- INITIALIZE HYPECUBE ------------------------ //
        // ------------------------------------------------------------------- //
        HyperCube hypercube(k, probes, M, &metric);
        // ------------------------------------------------------------------- //
        // ---------------------- INITIALIZE HYPECUBE ------------------------ //
        // ------------------------------------------------------------------- //

        // ------------------------------------------------------------------- //
        // ---------------------- METHOD INITIALIZATIONS --------------------- //
        // ------------------------------------------------------------------- //
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> range_approx;
        std::vector<std::pair<double, int>> nearest_approx_num;
        std::vector<std::pair<double, int>> range_approx_num;
        std::vector<std::pair<double, int>> nearest_exhaust;
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
        std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, 0);
        // Load the data to the hypercube
        hypercube.load_data(images);
        // images.insert(images.end(), queries.begin(), queries.end()); // Merge the two vectors of images so you can load them all at once
        // ------------------------------------------------------------------- //
        // ------------------------ OPEN AND LOAD INPUT ---------------------- //
        // ------------------------------------------------------------------- //
        int i = 0;
        while(i < QUERY_NUMBER  && i < (int)(queries.size())){
            nearest_approx_num.clear();
            range_approx_num.clear();
            // ------------------------------------------------------------------- //
            // ----------------------- APPROXIMATE NEAREST ----------------------- //
            // ------------------------------------------------------------------- //
            start = std::chrono::high_resolution_clock::now(); // Start the timer
            nearest_approx = hypercube.approximate_k_nearest_neighbors_return_images(queries[i], N); // Get the k approximate nearest vectors to the query
            end  = std::chrono::high_resolution_clock::now(); // End the timer 
            duration_approx = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
            // ------------------------------------------------------------------- //
            // ----------------------- APPROXIMATE NEAREST ----------------------- //
            // ------------------------------------------------------------------- //

            // ------------------------------------------------------------------- //
            // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
            // ------------------------------------------------------------------- //
            start = std::chrono::high_resolution_clock::now(); // Start the timer
            nearest_exhaust = exhaustive_nearest_neighbor_search(images, queries[i], N, &metric); // Get the k real nearest vectors to the query
            end  = std::chrono::high_resolution_clock::now(); // End the timer 
            duration_exhaust = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
            // ------------------------------------------------------------------- //
            // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
            // ------------------------------------------------------------------- //

            // ------------------------------------------------------------------- //
            // ------------------------ APPROXIMATE RANGE ------------------------ //
            // ------------------------------------------------------------------- //
            range_approx = hypercube.approximate_range_search_return_images(queries[i], R); // Get all the images that are in range R from the query
            // ------------------------------------------------------------------- //
            // ------------------------ APPROXIMATE RANGE ------------------------ //
            // ------------------------------------------------------------------- //

            // ------------------------------------------------------------------- //
            // ---------------------------- WRITES ------------------------------- //
            // ------------------------------------------------------------------- //
            for(int j = 0; j < (int)nearest_approx.size(); j++){
                auto temp = std::make_pair(nearest_approx[j].first, nearest_approx[j].second->get_number());
                nearest_approx_num.push_back(temp);
            }

            for(int j = 0; j < (int)range_approx.size(); j++){
                auto temp = std::make_pair(range_approx[j].first, range_approx[j].second->get_number());
                range_approx_num.push_back(temp);
            }
        
            write_approx_cube(queries[i], nearest_approx_num, nearest_exhaust, duration_approx, duration_exhaust, outputFile); 
            write_r_near(range_approx_num, R, outputFile);
            // ------------------------------------------------------------------- //
            // ---------------------------- WRITES ------------------------------- //
            // ------------------------------------------------------------------- //
            i++;
        }
    }
    else{
        printf("Give a path for the dataset file: ");
        std::cin >> inputFile;
        std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, 0);
        if(images.empty()){
            return -1;
        }

        HyperCube hypercube(k, probes, M, &metric);
        hypercube.load_data(images);

        bool wantMoreQueries = true;
        std::string answer;

        outputFile = fopen(outputFileName.c_str(), "w");
        if(outputFile == NULL){
            perror("Error opening output file");
            return -1;
        }
        do{
            printf("Give a path for the query file: ");
            std::cin >> queryFile;
            std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images(queryFile, 0);

            int i = 0;
            while(i < QUERY_NUMBER  && i < (int)(queries.size())){
                nearest_approx_num.clear();
                range_approx_num.clear();
                // ------------------------------------------------------------------- //
                // ----------------------- APPROXIMATE NEAREST ----------------------- //
                // ------------------------------------------------------------------- //
                start = std::chrono::high_resolution_clock::now(); // Start the timer
                nearest_approx = hypercube.approximate_k_nearest_neighbors_return_images(queries[i], N); // Get the k approximate nearest vectors to the query
                end  = std::chrono::high_resolution_clock::now(); // End the timer 
                duration_approx = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
                // ------------------------------------------------------------------- //
                // ----------------------- APPROXIMATE NEAREST ----------------------- //
                // ------------------------------------------------------------------- //

                // ------------------------------------------------------------------- //
                // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
                // ------------------------------------------------------------------- //
                start = std::chrono::high_resolution_clock::now(); // Start the timer
                nearest_exhaust = exhaustive_nearest_neighbor_search(images, queries[i], N, &metric); // Get the k real nearest vectors to the query
                end  = std::chrono::high_resolution_clock::now(); // End the timer 
                duration_exhaust = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // Calculate the duration
                // ------------------------------------------------------------------- //
                // ----------------------- EXHAUSTIVE NEAREST ------------------------ //
                // ------------------------------------------------------------------- //

                // ------------------------------------------------------------------- //
                // ------------------------ APPROXIMATE RANGE ------------------------ //
                // ------------------------------------------------------------------- //
                range_approx = hypercube.approximate_range_search_return_images(queries[i], R); // Get all the images that are in range R from the query
                // ------------------------------------------------------------------- //
                // ------------------------ APPROXIMATE RANGE ------------------------ //
                // ------------------------------------------------------------------- //

                // ------------------------------------------------------------------- //
                // ---------------------------- WRITES ------------------------------- //
                // ------------------------------------------------------------------- //
                for(int j = 0; j < (int)nearest_approx.size(); j++){
                    auto temp = std::make_pair(nearest_approx[j].first, nearest_approx[j].second->get_number());
                    nearest_approx_num.push_back(temp);
                }

                for(int j = 0; j < (int)range_approx.size(); j++){
                    auto temp = std::make_pair(range_approx[j].first, range_approx[j].second->get_number());
                    range_approx_num.push_back(temp);
                }
            
                write_approx_cube(queries[i], nearest_approx_num, nearest_exhaust, duration_approx, duration_exhaust, outputFile); 
                write_r_near(range_approx_num, R, outputFile);
                // ------------------------------------------------------------------- //
                // ---------------------------- WRITES ------------------------------- //
                // ------------------------------------------------------------------- //
                i++;
            }
            if(!queries.empty()){ // If you managed to read the query file inform of the results
                printf("Results have been written in ./out/cube.out\n");
            }
            printf("Type 'yes' if you want to try another query file: \n");
            std::cin >> answer;
            if(answer == "y" || answer == "Y" || answer == "yes" || answer == "Yes"){
                wantMoreQueries = true;
            }
            else{
                wantMoreQueries = false;
            }
        }while(wantMoreQueries);
    }
    fclose(outputFile);
    return 0;
}