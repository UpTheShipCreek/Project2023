#ifndef APPROXIMATE_METHODS_H
#define APPROXIMATE_METHODS_H

#include "hashtable.h"

class ApproximateMethods{
    public:
    virtual void load_data(std::vector<std::shared_ptr<ImageVector>> images) = 0;
    virtual std::vector<std::pair<double, int>> approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest) = 0;
    virtual std::vector<std::pair<double, int>> approximate_range_search(std::shared_ptr<ImageVector> image, double r) = 0;
};

#endif