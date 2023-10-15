#ifndef  IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <vector>
#include <memory>
#include <queue>

#include "metrics.h"

class ImageVector{
    int Number; // The number of the image 0-59999
    std::vector<double> Coordinates;

    public:
    ImageVector(int number, std::vector<double> coordinates);
    int get_number();
    std::vector<double> get_coordinates();
};

std::vector<std::pair<double, int>> exhaustive_nearest_neighbor_search(std::vector<std::shared_ptr<ImageVector>> images, std::shared_ptr<ImageVector> image, int numberOfNearest);


#endif