#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <stdio.h>
#include <fstream>
#include <memory>

#include "image_util.h"


std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename);
void write_approx_exhaust(shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> approx, std::vector<std::pair<double, int>> exhaust, double tLSH, double tTrue);
void write_r_near(shared_ptr<ImageVector> query, std::vector<std::pair<double, int>> inRange);
#endif