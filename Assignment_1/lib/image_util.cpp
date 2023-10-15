#include "image_util.h"

ImageVector::ImageVector(int number, std::vector<double> coordinates){
    this->Number = number;
    this->Coordinates = coordinates;
}

int  ImageVector::get_number(){
    return this->Number;
}

std::vector<double>  ImageVector::get_coordinates(){
    return this->Coordinates;
}

std::vector<std::pair<double, int>> exhaustive_nearest_neighbor_search(std::vector<std::shared_ptr<ImageVector>> images, std::shared_ptr<ImageVector> image, int numberOfNearest){
    int i;
    double distance;

    // I will be using a priority queue again
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> nearest;

    std::vector<std::pair<double, int>> nearestImages;

    for(i = 0; i < (int)(images.size()); i++){
        if(images[i]->get_number() != image->get_number()){ // Ignore comparing it to itself
            distance = eucledian_distance(image->get_coordinates(), images[i]->get_coordinates());
            nearest.push(std::make_pair(distance, images[i]->get_number()));
            if ((int)(nearest.size()) > numberOfNearest){
                nearest.pop();
            }
        }
    }
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    return nearestImages;
}
