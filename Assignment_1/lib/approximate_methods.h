#ifndef APPROXIMATE_METHODS_H
#define APPROXIMATE_METHODS_H

#include "hashtable.h"

class ApproximateMethods{
    public:
    virtual std::vector<std::pair<double, int>> approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest) = 0;
    virtual std::vector<std::pair<double, int>> approximate_range_search(std::shared_ptr<ImageVector> image, double r) = 0;
};

#endif