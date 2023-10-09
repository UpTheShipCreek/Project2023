#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <climits>
#include <fstream>
#include <string>
#include <getopt.h>

using namespace std;


int main(int argc, char **argv){
    int k = 4, L = 5, N = 1, R = 10000;                               //Default values for LSH algorithm, if not altered by user
    int opt;
    extern char *optarg; 
    string givenInput, queryFile, givenOutput;
    int cmdNecessary = 0;

    while ((opt = getopt(argc, argv, "d:q:k:L:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
        switch (opt) {
            case 'd':                                                               //Files
                givenInput = optarg;
                cmdNecessary++;
                break;
            case 'q':
                queryFile = optarg;
                cmdNecessary++;
                break;
            case 'o':
                givenOutput = optarg;
                cmdNecessary++;
                break;
            case 'k':                                                               //Parameters
                k = atoi(optarg);
            case 'L':
                L = atoi(optarg);
            case 'N':
                N = atoi(optarg);
                break;
            case 'R':
                R = atoi(optarg);
                break;
        }
    }

    if(cmdNecessary != 3){
        cout << "Program execution requires an input file, output file and a query file.";
        exit(-1);
    }
    else{
        cout << "Program will proceed with values: k = " << k << ",L = " << L <<  "N = " << N << "R = " << R << "\n";
    }

    //Input files parcing
    // Open the MNIST dataset file
    std::ifstream file(givenInput, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open MNIST dataset file." << std::endl;
        return 1;
    }

    // Skip the header information (16 bytes)
    file.seekg(16);

    // Define a vector to store the normalized pixel values for one image
    std::vector<double> imagePixels(784);

    // Create a vector to store all images
    std::vector<std::vector<double>> allImages;

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

    // Now, 'allImages' contains all the images from the MNIST dataset
    // You can access each image as 'allImages[i]' where 'i' is the image index.
    if (allImages.size() > 1) {
        std::cout << "Pixel values of the second image (allImages[2]):" << std::endl;
        
        // Iterate through the pixel values and print them
        for (int i = 0; i < allImages[2].size(); ++i) {
            std::cout << allImages[2][i] << " ";
        }
        
        std::cout << std::endl;
    } else {
        std::cerr << "There are not enough images in 'allImages' to display allImages[1]." << std::endl;
    }

    return 0;
}