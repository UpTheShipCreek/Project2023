#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<double>> readMNISTImages(const std::string& filename);

#endif