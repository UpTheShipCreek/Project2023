#include "cluster.h"

Cluster::Cluster(std::shared_ptr<ImageVector> centroid){
    this->Centroid = centroid;
}
Cluster::Cluster(std::vector<std::shared_ptr<ImageVector>> points){
    this->Points = points;
}
Cluster::Cluster(std::shared_ptr<ImageVector> centroid, std::vector<std::shared_ptr<ImageVector>> points){
    this->Centroid = centroid;
    this->Points = points;
}

void Cluster::set_centroid(std::shared_ptr<ImageVector> centroid){
    this->Centroid = centroid;
}
void Cluster::add_point(std::shared_ptr<ImageVector> point){
    (this->Points).push_back(point);
}

// Centroid_[n+1] = (N/N+1) Centroid_[n] + newPoint/N+1
void Cluster::add_point_and_set_centroid(std::shared_ptr<ImageVector> point){
    (this->Points).push_back(point);
    double numberOfPoints = ((this->Points).size());
    double fraction = (numberOfPoints) / (numberOfPoints + 1);

    double temp;
    double newvalue;

    if(this->Centroid->get_number() != -1){ // If the centroid is not virtual we need to create another one in order not to change the coordinates of the actual dataset image
        std::shared_ptr<ImageVector> centroidCopy = std::make_shared<ImageVector>(-1, this->Centroid->get_coordinates());
        this->Centroid = centroidCopy;
    }
    for (int i = 0; i < (int)(this->Centroid)->get_coordinates().size(); i++){
        temp =  (this->Centroid)->get_coordinates()[i];
        newvalue = (fraction * temp) + (point->get_coordinates()[i] / (numberOfPoints + 1));
        
        (this->Centroid)->get_coordinates()[i] = newvalue;
    }
}
std::shared_ptr<ImageVector>& Cluster::get_centroid(){
    return this->Centroid;
}
std::vector<std::shared_ptr<ImageVector>>& Cluster::get_points(){
    return this->Points;
}
std::shared_ptr<ImageVector> Cluster::recalculate_centroid(){
    int j;
    int clusterSize = (int)(this->Points).size();

    if(clusterSize == 0){
        return this->Centroid;
    }
    std::vector<double> sum = (this->Points[0])->get_coordinates(); // Get the first point as initialization
    for(int i = 1; i < clusterSize; i++){ // Sum the vectors
        for(j = 0; j < (int)sum.size(); j++){ // Add each coordinate of the vectors
            sum[j] += (this->Points[i])->get_coordinates()[j];
        }
    }
    for(j = 0; j < (int)sum.size(); j++){ // Divide each coordinate by the number of points
        sum[j] /= clusterSize;
    }
    if(this->Centroid->get_number() != -1){ // If the centroid is not virtual we need to create another one in order not to change the coordinates of the actual dataset image
        std::shared_ptr<ImageVector> centroidCopy = std::make_shared<ImageVector>(-1, sum);
        this->Centroid = centroidCopy;
    } 
    else{
        this->Centroid->get_coordinates() = sum;
    }
    return this->Centroid;
}
