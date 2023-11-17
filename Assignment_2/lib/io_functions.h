#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <stdio.h>
#include <fstream>
#include <memory>
#include <cfloat>

#include "image_util.h"

std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename, int imagesAlreadyRead);
void write_results(
    int datasetSize,std::shared_ptr<ImageVector> query, std::vector<std::pair<double, 
    std::shared_ptr<ImageVector>>> approx,std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaust, 
    double tApproximate, double tTrue, FILE* outputFile);
#endif

