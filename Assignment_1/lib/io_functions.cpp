#include "io_functions.h" // includes <iostream>, <fstream>, and <vector>


ImageVector::ImageVector(int number, std::vector<double> coordinates){
    this->Number = number;
    this->Coordinates = coordinates;
}

int  ImageVector::get_number(){
    return this->Number;
}

std::vector<double>  ImageVector::get_coordinates(){
    return this->Coordinates;
}

// Need to make this work with the new ImageVector class
std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename) {
    int imageNumberCounter = 0;
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

    return allImages;
}
