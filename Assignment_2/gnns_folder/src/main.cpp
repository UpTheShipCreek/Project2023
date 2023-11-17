#include <chrono>
#include "graph.h"


// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(int argc, char *argv[]){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> true_nearest_approx;
    std::vector<std::shared_ptr<Neighbors>> nearest_approx_for_all;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto approxTime = end - start;
    auto exhaustTime = end - start;

    Random rand;

    //Default values
    int k = GRAPH_DEFAULT_K, E = GRAPH_DEFAULT_E, R = GRAPH_DEFAULT_R, N = DEFAULT_N;
    int methodChoice;                                                                   //For method choice we have 1=GNNS, 2=MRNG
    std::string inputFile, queryFile, outputFileName, methodChoice, userAnswer;
    int continueExec = 0;

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

    // std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

     while(continueExec){
        // Get queries
        std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images(inputFile, 0);
        // Get the dataset
        std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images(queryFile, (int)queries.size());

        // LSH Optimal L: 6 K: 4 Window: 1400 TableSize: 7500
        std::shared_ptr<LSH> lsh = std::make_shared<LSH>(6, 4, 1400, 7500, &metric);
        Graph genericGraph(images, &metric);
        genericGraph.initialize_neighbours_approximate_method(lsh, k);
        int const numberOfQueries = 1000;

        std::vector<int> randomRestarts = {120, 130, 140, 150};
        for(auto r: randomRestarts){
            double maxFactor = DBL_MIN;
            double factor;
            double sum = 0;

            for(int i = 0; i < numberOfQueries; i++){
                int randomIndex = rand.generate_int_uniform(0, (int)queries.size() - 1);
                // GNNS Optimal K: 50 E: 30 G: 10 
                start = std::chrono::high_resolution_clock::now();
                nearest_approx = genericGraph.k_nearest_neighbor_search(queries[randomIndex], r, GRAPH_DEFAULT_G, E, N);
                end = std::chrono::high_resolution_clock::now();
                approxTime = end - start;
                
                start = std::chrono::high_resolution_clock::now();
                true_nearest_approx = exhaustive_nearest_neighbor_search_return_images(images, queries[randomIndex], N, &metric);
                end = std::chrono::high_resolution_clock::now();
                exhaustTime = end - start;

                factor = nearest_approx[0].first / true_nearest_approx[0].first;
                if(factor > maxFactor){
                    maxFactor = factor;
                }
                sum += factor;
            }
            
            printf("RandomRestarts:%d MaxFactor:%f AverageFactor: %f ApproxAverage: %ld ExhaustAverage: %ld\n", r, maxFactor, sum/numberOfQueries, approxTime.count() / numberOfQueries, exhaustTime.count() / numberOfQueries);
            fflush(stdout);
        }
        printf("Would you like to re-run the program for a new dataset/query (every other parameter will remain the same)? Y/N\n");
        std::cin >> userAnswer;
        if(userAnswer == "Y" || userAnswer == "y"){
            printf("Please give the path of the new input file: ");
            std::cin >> inputFile;
            printf("Please give the path of the new query file: ");
            std::cin >> queryFile;
        }
        else{
            continueExec = 1;
        }
    }
}