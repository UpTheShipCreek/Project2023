#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <climits>
#include <string>
#include <getopt.h>
#include <time.h>

#include "io_functions.h"

using namespace std;

//The main function. Bet you didn't expect that!
int main(int argc, char **argv){
    int k = 14, M = 10, probes = 2, N = 1, R = 10000;                               //Default values for hypercube projection, if not altered by user
    int opt;
    extern char *optarg; 
    string givenInput, queryFile, givenOutput;
    int cmdNecessary = 0;
    char inputCharacters[17] = "d:q:k:M:p:o:N:R:";
    double sttime, endtime; 

    while ((opt = getopt(argc, argv, inputCharacters)) != -1){                   //Parse through (potential) command line arguments
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
            case 'M':
                M = atoi(optarg);
            case 'p':
                probes = atoi(optarg);
                break;
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
        cout << "Program will proceed with values: k = " << k << ",M = " << M << ",probes = " << probes << "N = " << N << "R = " << R << "\n";
    }

    sttime=((double) clock())/CLOCKS_PER_SEC;                   //Can be moved depending on whether they want us to calculate 

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

    //exec hypercube method
    
    endtime=((double) clock())/CLOCKS_PER_SEC;

    std::ofstream outputFile("CubeOutputFile.txt");

    // Check if the file is opened successfully
    if (!outputFile) {
        std::cerr << "Error opening the output file." << std::endl;
        return 1;
    }

    // Write the template to the file
    outputFile << "Query: " <<  << std::endl;
    for (int i = 1; i <= /*numofnearneighbours*/; i++) {
        outputFile << "Nearest neighbor-" << i << ": " << /*dataSetImageNumber*/ << std::endl;
        outputFile << "distanceHypercube :" << /* distanceLSH */ << std::endl;
    outputFile << "distanceTrue: " << /* distanceTrue */ << std::endl;
    }
    outputFile << "tHypercube: " << /* tLSH */ << std::endl;
    outputFile << "tTrue: " << endtime-sttime << std::endl;
    outputFile << "R-near neighbors:" << std::endl;
    for (int i = 0; i < /* numNearNeighbors */; i++) {
        outputFile << /* image_number */ << std::endl;
    }

    // Close the output file
    outputFile.close(); 



/*     // Now, 'allImages' contains all the images from the MNIST dataset
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
 */
    return 0;
}