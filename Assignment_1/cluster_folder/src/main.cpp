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
#include <string.h>

#include "random_functions.h"
#include "io_functions.h"
#include "kmeans.h"
#include "lsh.h"
#include "hypercube.h"

struct Config{
    int number_of_clusters;
    int number_of_vector_hash_tables;
    int number_of_vector_hash_functions;
    int max_number_M_hypercube;
    int number_of_hypercube_dimensions;
    int number_of_probes;
};

bool parse_config_file(const std::string& filename, std::shared_ptr<Config> config){
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
    config->number_of_clusters = configMap["number_of_clusters"];
    config->number_of_vector_hash_tables = configMap["number_of_vector_hash_tables"];
    config->number_of_vector_hash_functions = configMap["number_of_vector_hash_functions"];
    config->max_number_M_hypercube = configMap["max_number_M_hypercube"];
    config->number_of_hypercube_dimensions = configMap["number_of_hypercube_dimensions"];
    config->number_of_probes = configMap["number_of_probes"];

    return true;
}

int main(int argc, char **argv){
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //
    bool complete = false;
    std::string inputFile, configFile, outputFileName, methodChoice;
    int cmdNecessary = 0;

    Eucledean metric;
    std::vector<double> silhouette;

    enum methods {classic = 0, lsh, hypercube};
    methods method;
    // ------------------------------------------------------------------- //
    // --------------------- PROGRAM INITIALIZATIONS --------------------- //
    // ------------------------------------------------------------------- //


    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
     for (int i = 1; i < argc; i++){ // Start from 1 to skip the program name (argv[0])
        std::string arg = argv[i];

        if(arg == "-i") {
            if (i + 1 < argc) {
                inputFile = argv[i + 1];
                i++; // Skip the next argument
            }
            cmdNecessary++;
        } 
        else if (arg == "-c") {
            if (i + 1 < argc) {
                configFile = argv[i + 1];
                i++;
            }
            cmdNecessary++;
        } 
        else if (arg == "-o") {
            if (i + 1 < argc) {
                outputFileName = argv[i + 1];
                i++;
            }
            cmdNecessary++;
        } 
        else if (arg == "-m") {
            if (i + 1 < argc) {
                methodChoice = argv[i + 1];
                i++;
            }
            cmdNecessary++;
        } 
        else if (arg == "-complete") {
            complete = true;
        }
    }

    if(cmdNecessary != 4){
        printf("Program execution requires an input file, output file, a cluster config file and a method selection.\n");
        return -1;
    }
    else{
        if(methodChoice.compare("Classic") == 0)
            method = classic;
        else if(methodChoice.compare("LSH") == 0)
            method = lsh;
        else if(methodChoice.compare("Hypercube") == 0)
            method = hypercube;
        else{
            printf("Error: Method given is not supported or does not exist.\n");
            return -1;
        }
    }
    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //
    FILE* outputFile = fopen(outputFileName.c_str(), "w");
    if(outputFile == NULL){
        perror("Error opening output file");
        return -1;
    }
    // ------------------------------------------------------------------- //
    // -------------------------- OPEN OUTPUT FILE ----------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------------- READ DATASET ------------------------- //
    // ------------------------------------------------------------------- //
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, 0);
    if(images.empty()){
        printf("Error reading dataset file\n");
        return -1;
    }
    // ------------------------------------------------------------------- //
    // ---------------------------- READ DATASET ------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------------- READ CONFIG -------------------------- //
    // ------------------------------------------------------------------- //
    std::shared_ptr<Config> config = std::make_shared<Config>();
    if(!parse_config_file(configFile, config)){
        printf("Error parsing config file\n");
        return -1;
    }
    // ------------------------------------------------------------------- //
    // ---------------------------- READ CONFIG -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ----------------------- K-MEANS INITIALIZATION -------------------- //
    // ------------------------------------------------------------------- //
    std::shared_ptr<kMeans> kmeans = std::make_shared<kMeans>(config->number_of_clusters, images, &metric);
    // ------------------------------------------------------------------- //
    // ----------------------- K-MEANS INITIALIZATION -------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ----------------------- TIME INITIALIZATION ----------------------- //
    // ------------------------------------------------------------------- //
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // ------------------------------------------------------------------- //
    // ----------------------- TIME INITIALIZATION ----------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // --------------------------- METHOD CASES -------------------------- //
    // ------------------------------------------------------------------- //
    if(method == 0){
        // ------------------------------------------------------------------- //
        // -------------------------- CLASSIC -------------------------------- //
        // ------------------------------------------------------------------- //
        start = std::chrono::high_resolution_clock::now();
        kmeans->mac_queen_with_lloyds();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        silhouette = kmeans->silhouette();

        std::vector<std::shared_ptr<Cluster>> clusters = kmeans->get_clusters();

        write_clustering(method, clusters, duration, silhouette, outputFile);
        if(complete){
            write_clustering_complete(clusters, outputFile);
        }
        // ------------------------------------------------------------------- //
        // -------------------------- CLASSIC -------------------------------- //
        // ------------------------------------------------------------------- //
    }
    else if(method == 1){
        // ------------------------------------------------------------------- //
        // -------------------------- LSH ------------------------------------ //
        // ------------------------------------------------------------------- //
        std::shared_ptr<LSH> lsh = std::make_shared<LSH>(config->number_of_vector_hash_tables, config->number_of_vector_hash_functions, MODULO, LSH_TABLE_SIZE, &metric);
        lsh->load_data(images);

        start = std::chrono::high_resolution_clock::now();
        kmeans->mac_queen_with_reverse(lsh);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        silhouette = kmeans->silhouette();

        std::vector<std::shared_ptr<Cluster>> clusters = kmeans->get_clusters();

        write_clustering(method, clusters, duration, silhouette, outputFile);
        if(complete){
            write_clustering_complete(clusters, outputFile);
        }
        // ------------------------------------------------------------------- //
        // -------------------------- LSH ------------------------------------ //
        // ------------------------------------------------------------------- //
    }
    else if(method == 2){
        // ------------------------------------------------------------------- //
        // -------------------------- HYPERCUBE ------------------------------ //
        // ------------------------------------------------------------------- //
        std::shared_ptr<HyperCube> cube = std::make_shared<HyperCube>(config->number_of_hypercube_dimensions, config->number_of_probes, config->max_number_M_hypercube, &metric);
        cube->load_data(images);

        start = std::chrono::high_resolution_clock::now(); 
        kmeans->mac_queen_with_reverse(cube);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        silhouette = kmeans->silhouette();
        
        std::vector<std::shared_ptr<Cluster>> clusters = kmeans->get_clusters();

        write_clustering(method, clusters, duration, silhouette, outputFile);
        if(complete){
            write_clustering_complete(clusters, outputFile);
        }
        // ------------------------------------------------------------------- //
        // -------------------------- HYPERCUBE ------------------------------ //
        // ------------------------------------------------------------------- //
    }
    // ------------------------------------------------------------------- //
    // --------------------------- METHOD CASES -------------------------- //
    // ------------------------------------------------------------------- //

    // ------------------------------------------------------------------- //
    // ---------------------------- WRITES ------------------------------- //
    // ------------------------------------------------------------------- //
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