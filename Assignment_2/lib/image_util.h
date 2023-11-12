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
    std::vector<double>& get_coordinates();

    std::size_t hash() const;
    bool operator==(const ImageVector& other) const;
};

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustive_nearest_neighbor_search_return_images(std::vector<std::shared_ptr<ImageVector>> images, std::shared_ptr<ImageVector> image, int numberOfNearest, Metric* metric);
std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustive_range_search(std::vector<std::shared_ptr<ImageVector>> images, std::shared_ptr<ImageVector> image, double r, Metric* metric);

#endif