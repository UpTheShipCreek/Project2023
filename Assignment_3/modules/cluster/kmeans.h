#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <climits>
#include <chrono>
#include <unistd.h>
#include <cfloat>
#include <functional>
#include <map>

#include "random_functions.h"
#include "io_functions.h"
#include "metrics.h"
#include "approximate_methods.h"
#include "cluster.h"

#define NUMBER_OF_CLUSTERS_CONVERGENCE_PERCENTAGE_TOLERANCE 0.8 // If at 100% of the clusters are converged then we have converged
#define DISTANCE_DIFFERENCE_AS_MAX_PERCENTAGE_TOLERANCE 0.01 // If the change in the distance is less than 1% of the max distance between two points in our dataset then we have converged
#define CHANGE_OF_DISTANCE_DIFFERENCE_PERCENTAGE_TOLERANCE 0.95 // Taking into account the percentage of the change of the change of distance between two epochs
#define OSCILATION_TOLERANCE 0.005
#define LEAST_NUMBER_OF_EPOCHS 5

double round_up_to_nearest_order_of_magnitude(double number);

class kMeans{
    int K; // Number of clusters
    std::vector<std::shared_ptr<ImageVector>> Points;
    std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Cluster>> PointToClusterMap; // Usuful information to have handy
    std::vector<std::shared_ptr<Cluster>> Clusters;
    std::vector<double> CenterMass;

    Random R;
    using AssignmentFunction = std::function<void()>; // Define a function pointer type for the mac_queen method
    double MaxDist;
    Metric* Kmetric;

    public:
    kMeans(std::vector<std::shared_ptr<Cluster>> Clusters, std::map<std::shared_ptr<ImageVector>, std::shared_ptr<Cluster>> PointToClusterMap, Metric* metric);
    kMeans(int k, std::vector<std::shared_ptr<ImageVector>> points, Metric* metric);
    std::shared_ptr<Cluster> get_nearest_cluster(std::shared_ptr<ImageVector> point);
    std::vector<std::shared_ptr<ImageVector>> get_centroids();
    std::vector<std::shared_ptr<Cluster>>& get_clusters();

    // --------------------------------------------------------- //
    // --------------------- Old functions --------------------- // 
    // --------------------------------------------------------- //
    void lloyds_assigment();
    void reverse_assignment(std::shared_ptr<ApproximateMethods> method);
    void traditional_convergence_algorithm(AssignmentFunction assignment);
    // --------------------------------------------------------- //
    // --------------------- Old functions --------------------- // 
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    // ------------------------- Methods ----------------------- // 
    // --------------------------------------------------------- //
    void mac_queen_with_lloyds();
    void mac_queen_with_reverse(std::shared_ptr<ApproximateMethods> method);
    // --------------------------------------------------------- //
    // ------------------------- Methods ----------------------- // 
    // --------------------------------------------------------- //

    std::shared_ptr<Cluster> get_nearest_cluster_excluding_the_assigned_one(std::shared_ptr<ImageVector> point);
    std::vector<double> silhouette();
};

#endif