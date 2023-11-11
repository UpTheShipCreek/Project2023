#include "graph.h"

// Because of how LSH works, the graph we end up having is a directed graph (i.e. the edges are not symmetric)

int main(void){
    Eucledean metric;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearest_approx;
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> lsh_nearest_approx;
    std::vector<std::shared_ptr<Neighbors>> nearest_approx_for_all;

    // std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>();

    // Get queries
    std::vector<std::shared_ptr<ImageVector>> queries = read_mnist_images("../Assignment_1/in/input.dat", 0);
    // Get the dataset
    std::vector<std::shared_ptr<ImageVector>> images = read_mnist_images("../Assignment_1/in/query.dat", (int)queries.size());


    // Load them into LSH
    LSH lsh(LSH_DEFAULT_L, LSH_DEFAULT_K, MODULO, LSH_TABLE_SIZE, &metric);
    lsh.load_data(images);
    printf("Creating the edge relations between the nodes/images...");
    fflush(stdout);
    for(auto& image : images){
        nearest_approx = lsh.approximate_k_nearest_neighbors_return_images(image, GRAPH_DEFAULT_K);
        // VERY IMPORTANT TO CREATE DIFFERENT POINTERS FOR EACH NEIGHBOR STRUCTURE
        std::shared_ptr<Neighbors> neighbors = std::make_shared<Neighbors>(); 
        for(auto& neighbor : nearest_approx){
            neighbors->push_back(neighbor.second);
        }
        nearest_approx_for_all.push_back(neighbors);
    }

    printf("Done\nCreating the graph...");
    fflush(stdout);
    Graph genericGraph(images, nearest_approx_for_all, &metric);
    printf("Done\nFinding nearest neighbors to the queries...\n");
    fflush(stdout);

    for(int i = 1; i <= 10; i++){
        printf("Query: %d\n", i);

        int greedySteps = 10;
        nearest_approx = genericGraph.k_nearest_neighbor_search(queries[i], GRAPH_DEFAULT_R, greedySteps, GRAPH_DEFAULT_E, GRAPH_DEFAULT_N);
        lsh_nearest_approx = lsh.approximate_k_nearest_neighbors_return_images(queries[i], GRAPH_DEFAULT_N);
        for(int i = 0; i < (int)nearest_approx.size(); i++){
            printf("LSH: %f Graph: %f\n", lsh_nearest_approx[i].first, nearest_approx[i].first);
        }
        printf("\n");
    }
    printf("Done\n");
    fflush(stdout);
}