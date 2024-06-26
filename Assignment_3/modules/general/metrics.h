#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath> // In case we need it for feature metrics

class Metric{
    public:
    virtual double calculate_distance(const std::vector<double>& p1, const std::vector<double>& p2) = 0;
};

class Eucledean : public Metric{
    public:
    double calculate_distance(const std::vector<double>& p1, const std::vector<double>& p2) override;
};

#endif