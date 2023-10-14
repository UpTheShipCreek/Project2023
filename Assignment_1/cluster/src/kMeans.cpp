#include <string>
#include <vector>
#include <algorithm>
#include <iterator>


std::vector<PointPtr> k_means(std::vector<PointPtr> pointsInput, int numCentroidPoints, int dimensions);
int choosePoint(std::vector<PointPtr> pointsInput, std::vector<double> distances);
double minCentroidDist(PointPtr p, std::vector<PointPtr> centroidPoints, int dimensions);
