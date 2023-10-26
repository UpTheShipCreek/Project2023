#include "io_functions.h" // includes <iostream>, <fstream>, and <vector>

// Need to make this work with the new ImageVector class
std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename, int imagesAlreadyRead){ // Need this so as the image numbers do not overlap
    int imageNumberCounter = imagesAlreadyRead + 1;
    std::shared_ptr<ImageVector> image;

    std::ifstream file(filename, std::ios::binary);
    std::vector<std::shared_ptr<ImageVector>> allImages;

    if(!file.is_open()){
        perror("Error while opening file");
        return allImages; // Return an empty vector in case of an error
    }

    // Skip the header information (16 bytes)
    file.seekg(16);

    // Define a vector to store the pixel values for one image
    std::vector<unsigned char> imagePixels(784);
    
    printf("Reading images... ");

    // Read and process each image entry
    while(file.read(reinterpret_cast<char*>(imagePixels.data()), 784)){
        // Convert pixel values from unsigned char to double and normalize
        std::vector<double> normalizedPixels(784);
        for (int i = 0; i < 784; i++){
            normalizedPixels[i] = static_cast<double>(imagePixels[i]); // Lets not normalize the pixels?
        }

        image = std::make_shared<ImageVector>(imageNumberCounter, normalizedPixels); // Ended up using smart pointers cause having to keep in mind to free that memory is not very good practice
        imageNumberCounter++;

        // Store the normalized pixel values in the container for all images
        allImages.push_back(image);
    }

    // Close the dataset file
    file.close();

    printf("Done\n");
    return allImages;
}


// Takes as input the query and two vectors, one of the approximate results and one of the exhaustive results
void write_approx_lsh(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tLSH, double tTrue, FILE* outputFile){
    fprintf(outputFile, "\nQuery: %d\n", query->get_number());
    for(int i = 0; i < (int)approx.size(); i++){
        fprintf(outputFile,"Nearest neighbor-%d: %d\n", i+1, approx[i].second);
        fprintf(outputFile,"distanceLSH: %f\n", approx[i].first);
        fprintf(outputFile,"distanceTrue: %f\n", exhaust[i].first);
    }
    fprintf(outputFile,"tLSH: %f\n", tLSH);
    fprintf(outputFile,"tTrue: %f\n", tTrue);
    fflush(outputFile);
}

void write_approx_cube(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tCube, double tTrue, FILE* outputFile){
    fprintf(outputFile, "\nQuery: %d\n", query->get_number());
    for(int i = 0; i < (int)approx.size(); i++){
        fprintf(outputFile,"Nearest neighbor-%d: %d\n", i+1, approx[i].second);
        fprintf(outputFile,"distanceHypercube: %f\n", approx[i].first);
        fprintf(outputFile,"distanceTrue: %f\n", exhaust[i].first);
    }
    fprintf(outputFile,"tHypercube: %f\n", tCube);
    fprintf(outputFile,"tTrue: %f\n", tTrue);
    fflush(outputFile);
}

// Write the results of the range search
void write_r_near(std::vector<std::pair<double, int>> inRange, int r, FILE* outputFile){
    fprintf(outputFile,"%d-near neighbors:\n", r);
    for(int i = 0; i < (int)inRange.size(); i++){
        fprintf(outputFile,"Nearest neighbor-%d: %d\n", i+1, inRange[i].second);
    }
    fflush(outputFile);
}

void write_clustering(int method, std::vector<std::shared_ptr<Cluster>> clusters, double clusteringTime, std::vector<double> silhouettes, FILE* outputFile){
    fprintf(outputFile, "Algorithm: ");
    if(method == 0){
        fprintf(outputFile, "Lloyd's\n");    
    }
    else if(method == 1){
        fprintf(outputFile, "Range Search LSH\n");
    }
    else if(method == 2){
        fprintf(outputFile, "Range Search Hypercube\n");
    }

    //CLUSTER-1 {size: <int>, centroid: πίνακας με τις συντεταγμένες του centroid}
    for(int i = 0; i < (int)clusters.size(); i++){
        fprintf(outputFile, "CLUSTER-%d {size: %d, centroid: ", i+1, (int)clusters[i]->get_points().size());
        std::shared_ptr<ImageVector> centroid = clusters[i]->get_centroid();
        for(int j = 0; j < (int)centroid->get_coordinates().size(); j++){
            fprintf(outputFile, "%f", centroid->get_coordinates()[j]);
            if(j != (int)centroid->get_coordinates().size() - 1){
                fprintf(outputFile, ", ");
            }
        }
        fprintf(outputFile, "}\n");
    }
    fprintf(outputFile, "clustering_time: %f\n", clusteringTime);
    fprintf(outputFile, "Silhouette: [");
    for(int i = 0; i < (int)silhouettes.size(); i++){
        fprintf(outputFile, "%f", silhouettes[i]);
        if(i != (int)silhouettes.size() - 1){
            fprintf(outputFile, ", ");
        }
    }
    fprintf(outputFile, "]\n");
    fflush(outputFile);
}

// CLUSTER-1 {centroid, image_numberA, ..., image_numberX
void write_clustering_complete(std::vector<std::shared_ptr<Cluster>> clusters, FILE* outputFile){
    int i, j;
    for(i = 0; i < (int)clusters.size(); i++){
        fprintf(outputFile, "CLUSTER-%d {centroid: ", i+1);
        if(clusters[i]->get_centroid()->get_number() == -1){
            fprintf(outputFile, "Virtual, ");
        }
        else{
            fprintf(outputFile, "%d", clusters[i]->get_centroid()->get_number());
        }
        int clusterSize = (int)clusters[i]->get_points().size();
        for(j = 0; j < clusterSize; j++){
            fprintf(outputFile, "%d", clusters[i]->get_points()[j]->get_number());
            if(i != clusterSize - 1){
                fprintf(outputFile, ", ");
            }
        }
        fprintf(outputFile, "}\n");
    }
        
}