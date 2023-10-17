#ifndef LSH_H
#define LSH_H

#include "hashtable.h"

class LSH{
    int K, L, M;
    std::vector<std::shared_ptr<HashTable>> Tables;

    public:
    LSH(int l, int k, int modulo, int tableSize);
    void load_data(std::vector<std::shared_ptr<ImageVector>> images);
    std::vector<std::pair<double, int>> approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest);
    std::vector<std::pair<double, int>> approximate_range_search(std::shared_ptr<ImageVector> image, double r);
};

#endif