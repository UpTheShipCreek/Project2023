#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <stdio.h>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"

#define DIMENSIONS 748
#define MODULO INT_MAX - 5
#define LSH_TABLE_SIZE 3750 // 60000/2^n
#define WINDOW 1500 // Test orders of magnitude
#define MEAN 0.0
#define STANDARD_DEVIATION 1.0

class HashFunction{
    public:
    virtual int evaluate_point(std::vector<double> p) = 0;
};

class hFunction{
    std::vector<double> V;
    double T;
    double  W = WINDOW;
    Random Rand;

    public:
    hFunction();
    double evaluate_point(std::vector<double> p);
};

class gFunction : public HashFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo
    std::vector<std::shared_ptr<hFunction>> H;
    std::vector<int> R; // The r values that will be used in the g function
    Random Rand; // Random generator

    public:
    gFunction(int k, int m);
    int evaluate_point(std::vector<double> p) override;
};

class fFunction{
    std::vector<std::pair<int, int>> KnownValues; // <h(p), f(h(p)) pairs
    Random Rand;

    public:
    int evaluate_point(int h_p); // Taking the projection of a point and projecting it into 0 or 1
};

class HashTable{
    int NumberOfBuckets;
    std::shared_ptr<HashFunction> HF;
    std::unordered_map<int, std::vector<std::shared_ptr<ImageVector>>> Table;
    std::unordered_map<int, int> NumberToId; // <number, id> pairs

    public:
    HashTable(int num, std::shared_ptr<HashFunction> hashfunction);
    bool same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2);
    void insert(std::shared_ptr<ImageVector> image);
    const std::vector<std::shared_ptr<ImageVector>>& get_bucket_from_image_vector(std::shared_ptr<ImageVector> image);

    std::pair<int, int> virtual_insert(std::shared_ptr<ImageVector> image);
    int get_image_id(std::shared_ptr<ImageVector> image);
    const std::vector<std::shared_ptr<ImageVector>>& get_bucket_from_bucket_id(int bucketId);
    int get_bucket_id_from_image_vector(std::shared_ptr<ImageVector> image);
};

#endif