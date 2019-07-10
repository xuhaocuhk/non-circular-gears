#include "geometry.h"
#include <cmath>

inline double square(const double &val) { return val * val; }

double length(const Point &p1, const Point &p2) {
    return sqrt(square(p2.x - p1.x) + square(p2.y - p1.y));
}

double perimeter(const Polygon &polygon) {
    double sum = 0;
    for (auto i = 0; i < polygon.size(); i++) {
        sum += length(polygon[i], at(polygon, i + 1));
    }
    return sum;
}
