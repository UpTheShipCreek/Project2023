#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <stdio.h>
#include <fstream>
#include <memory>

#include "image_util.h"
#include "cluster.h"


std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename, int imagesAlreadyRead);
void write_approx_lsh(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tLSH, double tTrue, FILE* outputFile);
void write_approx_cube(std::shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tCube, double tTrue, FILE* outputFile);
void write_r_near(std::vector<std::pair<double, int>> inRange, int r, FILE* outputFile);
void write_clustering(int method, std::vector<std::shared_ptr<Cluster>> clusters, double clusteringTime, std::vector<double> silhouettes, FILE* outputFile);
void write_clustering_complete(std::vector<std::shared_ptr<Cluster>> clusters, FILE* outputFile);
#endif

