#include "metrics.h"

double eucledian_distance(const std::vector<double>& p1, const std::vector<double>& p2){ // Eucledean distance function between two points in vector form
    double sum = 0.0;
    int size;

    if(p1.size() > p2.size()) size = p2.size();  // If the vectors are not equal, calculate the distance in regards to their common length
    else size = p1.size();

    for(int i = 0; i < size; i++){
        sum += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(sum);
}
