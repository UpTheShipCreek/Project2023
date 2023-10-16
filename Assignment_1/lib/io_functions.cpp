#include "io_functions.h" // includes <iostream>, <fstream>, and <vector>

// Need to make this work with the new ImageVector class
std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename, int imagesAlreadyRead){ // Need this so as the image numbers do not overlap
    int imageNumberCounter = imagesAlreadyRead + 1;
    std::shared_ptr<ImageVector> image;

    std::ifstream file(filename, std::ios::binary);
    std::vector<std::shared_ptr<ImageVector>> allImages;

    if (!file.is_open()){
        perror("Error while opening file");
        return allImages; // Return an empty vector in case of an error
    }

    // Skip the header information (16 bytes)
    file.seekg(16);

    // Define a vector to store the pixel values for one image
    std::vector<unsigned char> imagePixels(784);
    
    printf("Reading images... ");

    // Read and process each image entry
    while (file.read(reinterpret_cast<char*>(imagePixels.data()), 784)) {
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
void write_approx_exhaust(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tLSH, double tTrue, FILE* outputFile){
    fprintf(outputFile, "Query: %d\n", query->get_number());
    for(int i = 0; i < (int)approx.size(); i++){
        fprintf(outputFile,"Nearest neighbor-%d: %d\n", i+1, approx[i].second);
        fprintf(outputFile,"distanceLSH: %f\n", approx[i].first);
        fprintf(outputFile,"distanceTrue: %f\n", exhaust[i].first);
    }
    fprintf(outputFile,"tLSH: %f\n", tLSH);
    fprintf(outputFile,"tTrue: %f\n\n", tTrue);
}
// Write the results of the range search
void write_r_near(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> inRange, FILE* outputFile){
    fprintf(outputFile,"%d-near neighbors:\n", query->get_number());
    for(int i = 0; i < (int)inRange.size(); i++){
        fprintf(outputFile,"Nearest neighbor-%d: %d\n\n", i+1, inRange[i].second);
    }
}