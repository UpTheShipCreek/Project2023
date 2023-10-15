#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <stdio.h>
#include <fstream>
#include <memory>

#include "image_util.h"


std::vector<std::shared_ptr<ImageVector>> read_mnist_images(const std::string& filename);

#endif