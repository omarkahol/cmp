#include "klassifier.h"

void cmp::classifier::KNN::compute(const std::vector<Eigen::VectorXd> &points, size_t k)
{
    // Set the number of points and the number of neighbors
    nPoints_ = points.size();
    kNearestValue_ = k;

    // Initialize the neighbors
    neighbours_ = std::vector<std::vector<size_t>>(nPoints_, std::vector<size_t>(kNearestValue_, 0));

    // Iterate through the points
    for (size_t i =0; i < nPoints_; i++) {

        // Current point
        Eigen::VectorXd point = points[i];
        
        // Create a linked list for easy storing of the neighbors
        std::list<std::pair<double, size_t>> neighbors;

        for (size_t j = 0; j < nPoints_; j++) {

            // Skip the current point because it is at a distance of 0 from itself
            if (i == j) {
                continue; // Go to the next point
            }
            
            // Compute the distance
            double dist = (point - points[j]).norm();

            // Check if the list is empty
            // If it is, just add the point
            if (neighbors.empty()) {
                neighbors.push_back(std::make_pair(dist, j));
                continue; // Go to the next point
            }

            // Now we check if we can insert the point in the list
            bool inserted = false; // Flag to check if the point was inserted
            for (auto it = neighbors.begin(); it != neighbors.end(); it++) {
                
                // If point j is closer than the current point, insert it
                if (dist < it->first) {
                    neighbors.insert(it, std::make_pair(dist, j)); // Insert it before the current point
                    inserted = true;
                    break; // Go to the next point (exit the iterator loop)
                }
            }
            
            // If the point was not inserted, check if the list is not full
            if (neighbors.size() < kNearestValue_ && !inserted) {
                neighbors.push_back(std::make_pair(dist, j)); // Add the point to the end of the list
                continue;
            }

            // Check if the list is full
            // If the list is full, remove the last element
            if (neighbors.size() > kNearestValue_) {
                neighbors.pop_back();
                continue;
            }
        }

        // Save the neighbors
        auto iterator = neighbors.begin();
        for (size_t j = 0; j < kNearestValue_; j++) {
            neighbours_[i][j] = iterator->second;
            iterator++;
        }
    }
}

const std::vector<size_t> &cmp::classifier::KNN::operator[](size_t i) const
{
    return neighbours_[i];
}

size_t cmp::classifier::KNN::nPoints() const
{
    return nPoints_;
}

size_t cmp::classifier::KNN::k() const
{
    return kNearestValue_;
}
