#ifndef CLUSTER_H
#define CLUSTER_H

#include "image_util.h"

class Cluster{
    std::shared_ptr<ImageVector> Centroid;
    std::vector<std::shared_ptr<ImageVector>> Points;

    public:
    Cluster();
    Cluster(std::shared_ptr<ImageVector> centroid);
    Cluster(std::vector<std::shared_ptr<ImageVector>> points);
    Cluster(std::shared_ptr<ImageVector> centroid, std::vector<std::shared_ptr<ImageVector>> points);
    void set_centroid(std::shared_ptr<ImageVector> centroid);
    void add_point(std::shared_ptr<ImageVector> point);
    // Centroid_[n+1] = (N/N+1) Centroid_[n] + newPoint/N+1
    void add_point_and_set_centroid(std::shared_ptr<ImageVector> point);
    void remove_point_and_set_centroid(std::shared_ptr<ImageVector> point);
    std::shared_ptr<ImageVector>& get_centroid();
    std::vector<std::shared_ptr<ImageVector>>& get_points();
    std::shared_ptr<ImageVector> recalculate_centroid();
};
#endif


