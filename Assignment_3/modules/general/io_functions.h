#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <stdio.h>
#include <fstream>
#include <memory>
#include <cfloat>
#include <netinet/in.h>

#include "image_util.h"

class HeaderInfo{
    int32_t numberOfImages;
    int32_t numberOfRows;
    int32_t numberOfColumns;

    public:
    HeaderInfo(int32_t numberOfImages, int32_t numberOfRows, int32_t numberOfColumns);
    int32_t get_numberOfImages();
    int32_t get_numberOfRows();
    int32_t get_numberOfColumns();
    bool operator==(const HeaderInfo& other) const;
};

std::pair<std::shared_ptr<HeaderInfo>, std::vector<std::shared_ptr<ImageVector>>> read_mnist_images(const std::string& filename, int imagesAlreadyRead);
void write_results(
    int datasetSize,std::shared_ptr<ImageVector> query, std::vector<std::pair<double, 
    std::shared_ptr<ImageVector>>> approx,std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaust, 
    FILE* outputFile);
#endif

