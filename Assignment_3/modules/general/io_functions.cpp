#include "io_functions.h" // includes <iostream>, <fstream>, and <vector>


HeaderInfo::HeaderInfo(int32_t numberOfImages, int32_t numberOfRows, int32_t numberOfColumns){
    this->numberOfImages = numberOfImages;
    this->numberOfRows = numberOfRows;
    this->numberOfColumns = numberOfColumns;
}

int32_t HeaderInfo::get_numberOfImages(){
    return numberOfImages;
}
int32_t HeaderInfo::get_numberOfRows(){
    return numberOfRows;
}
int32_t HeaderInfo::get_numberOfColumns(){
    return numberOfColumns;
}

bool HeaderInfo::operator==(const HeaderInfo& other) const{
    return (numberOfRows == other.numberOfRows) && (numberOfColumns == other.numberOfColumns);
}

// Need to make this work with the new ImageVector class
std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> read_mnist_images(const std::string& filename, int imagesAlreadyRead){ // Need this so as the image numbers do not overlap
    int imageNumberCounter = imagesAlreadyRead + 1;
    std::shared_ptr<ImageVector> image;

    std::ifstream file(filename, std::ios::binary);
    std::vector<std::shared_ptr<ImageVector>> allImages;
    std::shared_ptr<HeaderInfo> headerInfo;

    if(!file.is_open()){
        perror("Error while opening file");
        return {headerInfo, allImages}; // Return an empty vector in case of an error
    }

    // Skip the header information (4 bytes)
    file.seekg(4);

    // Read the number of images
    uint32_t numberOfImages;
    file.read(reinterpret_cast<char*>(&numberOfImages), sizeof(numberOfImages));
    numberOfImages = ntohl(numberOfImages); 
    printf("Number of images: %d\n", numberOfImages);

    // Read the number of rows
    uint32_t numberOfRows;
    file.read(reinterpret_cast<char*>(&numberOfRows), sizeof(numberOfRows));
    numberOfRows = ntohl(numberOfRows); // Convert to little endian
    printf("Number of rows: %d\n", numberOfRows);

    // Read the number of columns
    uint32_t numberOfColumns;
    file.read(reinterpret_cast<char*>(&numberOfColumns), sizeof(numberOfColumns));
    numberOfColumns = ntohl(numberOfColumns); 
    printf("Number of columns: %d\n", numberOfColumns);

    headerInfo = std::make_shared<HeaderInfo>(numberOfImages, numberOfRows, numberOfColumns);

    int numberOfPixels = numberOfRows * numberOfColumns;

    // Define a vector to store the pixel values for one image
    std::vector<unsigned char> imagePixels(numberOfPixels);
    
    printf("Reading images... ");

    // Read and process each image entry
    while(file.read(reinterpret_cast<char*>(imagePixels.data()), numberOfPixels)){
        // Convert pixel values from unsigned char to double and normalize
        std::vector<double> normalizedPixels(numberOfPixels);
        for (int i = 0; i < numberOfPixels; i++){
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
    return {headerInfo, allImages};
}

void write_results(
    int datasetSize,
    std::shared_ptr<ImageVector> query, 
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> approx,
    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaust,
    FILE* outputFile
    ){
        double approxDistance, exhaustDistance;

        if (!outputFile) {
            printf("Error opening output file. Exiting...\n");
            return;
        }

        fprintf(outputFile, "Query: %d\n", query->get_number() - datasetSize);
        for(int i = 0; i < (int)approx.size(); i++){
            fprintf(outputFile, "Nearest neighbor-%d: %d\n", i+1, approx[i].second->get_number());
            approxDistance = approx[i].first;
            exhaustDistance = exhaust[i].first;
            fprintf(outputFile, "distanceApproximate: %f\n", approxDistance);
            fprintf(outputFile, "distanceTrue: %f\n", exhaustDistance);
        }
        fprintf(outputFile, "\n");
        fflush(outputFile);
    }