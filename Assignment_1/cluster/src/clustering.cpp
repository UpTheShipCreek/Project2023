#include <string>
#include <vector>

typedef struct PointStruct *PointPtr;

typedef struct PointStruct
{
    std::string id;
    std::vector<double> coords;
} Point;

typedef struct ClusterStruct *ClusterPtr;
typedef struct ClusterStruct
{
    PointPtr centroidPoint;
    std::vector<PointPtr> points;
    int size;
    double silhouette;
} Cluster;

typedef struct ClusterDataStruct
{

    std::string inputFileName,
        configFileName,
        outputFileName,
        methodName;

    int method;

    bool complete;

    int number_of_clusters,
        number_of_vector_hash_tables,
        number_of_vector_hash_functions,
        max_number_M_hypercube,
        number_of_hypercube_dimensions,
        number_of_probes;

    int numberOfInputPoints;
    int dimension;

} inputData;