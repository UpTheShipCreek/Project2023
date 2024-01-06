#include "image_util.h"

// The dataset is fine but then the index of the queryset is wrong, it starts from 60000 but reduced[60000] is out of bounds for the queryset
SpaceCorrespondace::SpaceCorrespondace(std::vector<std::shared_ptr<ImageVector>> initial){
    int number;

    for(auto& element : initial){
        number = element->get_number();
        InitialSpace[number] = element->get_coordinates();
    }
}
std::vector<double> SpaceCorrespondace::get_initial(int number){
    return InitialSpace[number];
}


ImageVector::ImageVector(int number, std::vector<double> coordinates){
    this->Number = number;
    this->Coordinates = coordinates;
}

int  ImageVector::get_number(){
    return this->Number;
}

std::vector<double>& ImageVector::get_coordinates(){
    return this->Coordinates;
}

std::size_t ImageVector::hash() const{
    return std::hash<int>()(Number);
}

// Implement custom equality operator for ImageVector
bool ImageVector::operator==(const ImageVector& other) const{
    return Coordinates == other.Coordinates;
}

std::vector<std::pair<double, int>> exhaustive_nearest_neighbor_search(
    std::vector<std::shared_ptr<ImageVector>> images, 
    std::shared_ptr<ImageVector> image, 
    int numberOfNearest,
    Metric* metric){

    int i;
    double distance;

    // I will be using a priority queue again
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>> nearest;

    std::vector<std::pair<double, int>> nearestImages;

    for(i = 0; i < (int)(images.size()); i++){
        if(images[i] != image){ // Ignore comparing it to itself
            distance = metric->calculate_distance(image->get_coordinates(), images[i]->get_coordinates());
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
    std::vector<std::pair<double,int>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;
}

std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustive_nearest_neighbor_search_return_images(
    std::vector<std::shared_ptr<ImageVector>> images, 
    std::shared_ptr<ImageVector> image, 
    int numberOfNearest,
    Metric* metric){

    int i;
    double distance;

    // I will be using a priority queue again
    std::priority_queue<std::pair<double, std::shared_ptr<ImageVector>>, std::vector<std::pair<double, std::shared_ptr<ImageVector>>>, std::less<std::pair<double, std::shared_ptr<ImageVector>>>> nearest;

    std::vector<std::pair<double, std::shared_ptr<ImageVector>>> nearestImages;

    for(i = 0; i < (int)(images.size()); i++){
        if(images[i] != image){ // Ignore comparing it to itself
            distance = metric->calculate_distance(image->get_coordinates(), images[i]->get_coordinates());
            nearest.push(std::make_pair(distance, images[i]));
            if ((int)(nearest.size()) > numberOfNearest){
                nearest.pop();
            }
        }
    }
    while (!nearest.empty()){
        nearestImages.push_back(nearest.top());
        nearest.pop();
    }
    std::vector<std::pair<double,std::shared_ptr<ImageVector>>> reversed(nearestImages.rbegin(), nearestImages.rend()); // Our vector is in reverse order so we need to reverse it
    return reversed;
}


std::vector<std::pair<double, std::shared_ptr<ImageVector>>> exhaustive_range_search(
    std::vector<std::shared_ptr<ImageVector>> images, 
    std::shared_ptr<ImageVector> image, 
    double r,
    Metric* metric){
        int i;
        double distance;

        // The returned vector
        std::vector<std::pair<double, std::shared_ptr<ImageVector>>> inRangeImages;

        for(i = 0; i < (int)(images.size()); i++){
            if(images[i]->get_number() != image->get_number()){ // Ignore comparing to itself
                distance = metric->calculate_distance(image->get_coordinates(), images[i]->get_coordinates());
                if(distance <= r){
                    inRangeImages.push_back(std::make_pair(distance, images[i]));
                }
            }
        }
    return inRangeImages;
    }