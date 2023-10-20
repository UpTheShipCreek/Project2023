#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <unordered_set>

#include "hashtable.h"
#include "approximate_methods.h"

int hamming_distance(int x, int y);
std::vector<int> find_all_with_hamming_distance_one(int input, int dimensions);
std::vector<int> get_probes(int number, int numberOfProbes, int dimensions);

class HypercubeHashFunction : public HashFunction{
    int K; // d' i.e. the dimension of the hypercube on which the points will be projected to
    std::vector<std::shared_ptr<fFunction>> F; // The f functions
    std::vector<std::shared_ptr<hFunction>> H; // The h functions

    public:
    HypercubeHashFunction(int k);
    int evaluate_point(std::vector<double> p) override;
};

class HyperCube : public ApproximateMethods{
    int K, Probes, M;
    std::shared_ptr<HashTable> Table;

    public:
    HyperCube(int dimensions, int probes, int numberOfElementsToCheck);
    void load_data(std::vector<std::shared_ptr<ImageVector>> images);
    std::vector<std::pair<double, int>> approximate_k_nearest_neighbors(std::shared_ptr<ImageVector> image, int numberOfNearest);
    std::vector<std::pair<double, int>> approximate_range_search(std::shared_ptr<ImageVector> image, double r);
};

#endif