#include "graph.h"

// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(int argc, char *argv[]){
    //Default values
    int k = GRAPH_DEFAULT_K, E = GRAPH_DEFAULT_E, R = GRAPH_DEFAULT_R, N = GRAPH_DEFAULT_N;
    int methodChoice;
    std::string inputFile, queryFile, outputFileName, methodChoice;
    
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> lsh_nearest_approx;
    std::vector<std::shared_ptr<Neighbors>> nearest_approx_for_all;

    // std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //
    int cmdNecessary;
    for (int i = 1; i < argc; i++){ // Start from 1 to skip the program name (argv[0])
        std::string arg = argv[i];

        if(arg == "-i"){
            if (i + 1 < argc){
                inputFile = argv[i + 1];
                i++; // Skip the next argument
            }
            cmdNecessary++;
        } 
        else if (arg == "-q"){
            if (i + 1 < argc){
                queryFile = argv[i + 1];
                i++;    
            }
            cmdNecessary++;
        } 
        else if (arg == "-o"){
            if (i + 1 < argc){
                outputFileName = argv[i + 1];
                i++;
            }
            cmdNecessary++;
        } 
        else if (arg == "-m"){
            if (i + 1 < argc){
                methodChoice = atoi(argv[i + 1]);
                i++;
            }
            cmdNecessary++;
        } 
        else if (arg == "-k"){
           if (i + 1 < argc){
                k = atoi(argv[i + 1]);
                i++;
            }
        }
        else if (arg == "-E"){
           if (i + 1 < argc){
                E = atoi(argv[i + 1]);
                i++;
            }
        }
        else if (arg == "-R"){
           if (i + 1 < argc){
                R = atoi(argv[i + 1]);
                i++;
            }
        }
        else if (arg == "-N"){
           if (i + 1 < argc){
                N = atoi(argv[i + 1]);
                i++;
            }
        }
    }
    if(cmdNecessary != 4){
        printf("Program execution requires an input file, output file, a query file and a method selection.\n");
        return -1;
    }
    else{
        if(methodChoice != 1 && methodChoice != 2){
            printf("Error: Method input can only be 1 for GNNS or 2 for MNRG.\n");
            return -1;
        }
    }
    // ------------------------------------------------------------------- //
    // -------------------------- INPUT PARSING -------------------------- //
    // ------------------------------------------------------------------- //


    // Get queries
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/query.dat", (int)queries.size());

    std::shared_ptr<LSH> lsh = std::make_shared<LSH>(LSH_DEFAULT_L, LSH_DEFAULT_K, MODULO, LSH_TABLE_SIZE, &metric);
    Graph genericGraph(images, &metric);
    genericGraph.initialize_neighbours_approximate_method(lsh);

    for(int i = 1; i <= 10; i++){
        printf("Query: %d\n", i);

        int greedySteps = 10;
        nearest_approx = genericGraph.k_nearest_neighbor_search(queries[i], R, greedySteps, E, N);
        lsh_nearest_approx = lsh->approximate_k_nearest_neighbors_return_images(queries[i], N);
        for(int i = 0; i < (int)nearest_approx.size(); i++){
            printf("LSH: %f Graph: %f\n", lsh_nearest_approx[i].first, nearest_approx[i].first);
        }
        printf("\n");
    }
    printf("Done\n");
    fflush(stdout);
}