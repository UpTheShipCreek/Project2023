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

void  ImageVector::assign_id(int id){
    this->Id = id;
}

int  ImageVector::get_id(){
    return this->Id;
}


// Need to make this work with the new ImageVector class
std::vector<ImageVector*> read_mnist_images(const std::string& filename) {
    int imageNumberCounter = 0;
    ImageVector* image;

    std::ifstream file(filename, std::ios::binary);
    std::vector<ImageVector*> allImages;

    if (!file.is_open()) {
        std::cerr << "Failed to open MNIST dataset file." << std::endl;
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
        for (int i = 0; i < 784; ++i) {
            normalizedPixels[i] = static_cast<double>(imagePixels[i]) / 1.0; // Lets not normalize the pixels?
        }

        image = new ImageVector(imageNumberCounter, normalizedPixels); // Need to free them, maybe in the main
        imageNumberCounter++;

        // Store the normalized pixel values in the container for all images
        allImages.push_back(image);
    }

    // Close the dataset file
    file.close();

    return allImages;
}
