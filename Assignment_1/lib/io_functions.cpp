#include "io_functions.h" // includes <iostream>, <fstream>, and <vector>

std::vector<std::vector<double>> readMNISTImages(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<double>> allImages;

    if (!file.is_open()) {
        std::cerr << "Failed to open MNIST dataset file." << std::endl;
        return allImages; // Return an empty vector in case of an error
    }

    // Skip the header information (16 bytes)
    file.seekg(16);

    // Define a vector to store the normalized pixel values for one image
    std::vector<double> imagePixels(784);

    // Read and process each image entry
    while (file.read(reinterpret_cast<char*>(imagePixels.data()), 784)) {
        // Normalize the pixel values by dividing by 255.0
        for (int i = 0; i < 784; ++i) {
            imagePixels[i] /= 255.0;
        }

        // Store the normalized pixel values in the container for all images
        allImages.push_back(imagePixels);
    }

    // Close the dataset file
    file.close();

    return allImages;
}