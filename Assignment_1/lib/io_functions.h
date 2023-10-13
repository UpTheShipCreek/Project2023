#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <vector>


class ImageVector{
    int Number; // The number of the image 0-59999
    //int Id = -1; // The id of the image needed for the querying trick, initialized as -1 so as to check if it has been assigned
    std::vector<double> Coordinates;

    public:
    ImageVector(int number, std::vector<double> coordinates);
    int get_number();
    std::vector<double> get_coordinates();
    // void assign_id(int id); // This Id should be different for every bucket but the way I have it implemented it is the same for the same image in different buckets
    // int get_id();
};

std::vector<ImageVector*> read_mnist_images(const std::string& filename);

#endif