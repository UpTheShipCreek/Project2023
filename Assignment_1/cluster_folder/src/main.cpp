#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <getopt.h>

#include "random_functions.h"
#include "io_functions.h"

using namespace std;

struct Config {
    int number_of_clusters;
    int number_of_vector_hash_tables;
    int number_of_vector_hash_functions;
    int max_number_M_hypercube;
    int number_of_hypercube_dimensions;
    int number_of_probes;
};

bool parseConfigFile(const std::string& filename, Config& config){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Unable to open the config file." << std::endl;
        return false;
    }

    std::map<std::string, int> configMap;

    std::string line;
    while(std::getline(file, line)){
        size_t pos = line.find(':');
        if(pos != std::string::npos){
            std::string key = line.substr(0, pos);
            int value = std::stoi(line.substr(pos + 1));
            configMap[key] = value;
        }
    }

    file.close();

    // Populate the Config struct with parsed values
    config.number_of_clusters = configMap["number_of_clusters"];
    config.number_of_vector_hash_tables = configMap["number_of_vector_hash_tables"];
    config.number_of_vector_hash_functions = configMap["number_of_vector_hash_functions"];
    config.max_number_M_hypercube = configMap["max_number_M_hypercube"];
    config.number_of_hypercube_dimensions = configMap["number_of_hypercube_dimensions"];
    config.number_of_probes = configMap["number_of_probes"];

    return true;
}

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    int kLSH = 4, L = 5, N = 1;                                             // Default values for lsh
    int kHC = 3, M = 10, probes = 2;                                       //Default values for hypercube
    int completeF;
    double R = 2000.0;                           
    int opt;
    extern char *optarg; 
    std::string inputFile, configFile, outputFileName, methodChoice;
    int cmdNecessary = 0;
    enum methods {classic = 0, lsh, hypercube};
    methods method;
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //


    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
    while ((opt = getopt(argc, argv, "i:c:o:complete:m:")) != -1){                   //Parse through (potential) command line arguments
        switch (opt) {
            case 'i':                                                               //Files
                inputFile = optarg;
                cmdNecessary++;
                break;
            case 'c':
                configFile = optarg;
                cmdNecessary++;
                break;
            case 'o':
                outputFileName = optarg;
                cmdNecessary++;
                break;
            case 'complete':                                                               //Parameters
                completeF = 1;
                break;
            case 'm':
                methodChoice = optarg;
                cmdNecessary++;
                break;  
        }
    }

    if(cmdNecessary != 4){
        printf("Program execution requires an input file, output file, a cluster config file and a method selection.");
        return -1;
    }
    else{
        if (methodChoice.compare("Classic") == 0)
            method = classic;
        else if (methodChoice.compare("LSH") == 0)
            method = lsh;
        else if (methodChoice.compare("Hypercube") == 0)
            method = hypercube;
        else {
            cout << "Error: Method given is not supported or does not exist." << endl;
            exit(-1);
        }
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
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_cluster = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // ------------------------------------------------------------------- //
    // ------------------------- INITIALIZE TIME ------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------------- WRITES ------------------------------- //
    // ------------------------------------------------------------------- //
    write_clustering(/*other arguments*/ method, outputFile);
    // ------------------------------------------------------------------- //
    // ---------------------------- WRITES ------------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //
    fclose(outputFile);
    // ------------------------------------------------------------------- //
    // -------------------------- CLOSE OUTPUT FILE ---------------------- //
    // ------------------------------------------------------------------- //

    return 0;
}