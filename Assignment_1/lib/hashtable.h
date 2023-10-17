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

class hFunction{
    std::vector<double> V;
    double T;
    double  W = WINDOW;
    Random Rand;

    public:
    hFunction();
    double evaluate_point(std::vector<double> p);
};

class gFunction{
    int K; // Number of hi functions that a g will be combining
    int M; // The modulo
    std::vector<std::shared_ptr<hFunction>> H;
    std::vector<int> R; // The r values that will be used in the g function
    Random Rand; // Random generator

    public:
    gFunction(int k, int m);
    int evaluate_point(std::vector<double> p);
};

class HashTable{
    int NumberOfBuckets;
    std::shared_ptr<gFunction> HashFunction;
    std::unordered_map<int, std::vector<std::shared_ptr<ImageVector>>> Table;
    std::unordered_map<int, int> NumberToId; // <number, id> pairs

    public:
    HashTable(int num, int k, int m);
    bool same_id(std::shared_ptr<ImageVector> image1, std::shared_ptr<ImageVector> image2);
    void insert(std::shared_ptr<ImageVector> image);
    const std::vector<std::shared_ptr<ImageVector>>& get_bucket_of_image(std::shared_ptr<ImageVector> image);
};

#endif