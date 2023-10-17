#ifndef HYPERCUBE_H
#define HYPERCUBE_H

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
#endif  