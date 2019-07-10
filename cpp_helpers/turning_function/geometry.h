#pragma once

#include <vector>

struct Point {
    double x, y;
};
typedef std::vector<Point> Polygon;

const double EPS = 1e-5;

inline const Point &at(const Polygon &polygon, unsigned index){
    index %= polygon.size();
    return polygon[index];
}

double length(const Point &p1, const Point &p2);
double perimeter(const Polygon &polygon);
