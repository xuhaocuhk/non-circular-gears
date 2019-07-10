#include "geometry.h"
#include <cmath>
#include <cassert>

double perimeter(const EdgePolygon &polygon) {
    double sum = 0;
    for (const auto &edge:polygon) sum += edge.length();
    return sum;
}

double turningAngle(const EdgePolygon &polygon, int index) {
    const auto &self = at(polygon, index);
    const auto &next = at(polygon, index + 1);
    auto angle = acos(dot(self, next) / self.length() / next.length());
    if (cross(self, next) < 0) angle = -angle;
    return angle;
}

EdgePolygon toEdgePolygon(const PointPolygon &polygon) {
    EdgePolygon edgePolygon;
    for (int i = 0; i < polygon.size(); i++) {
        edgePolygon.push_back(at(polygon, i + 1) - polygon[i]);
    }
    assert(checkEdgePolygon(edgePolygon));
    return edgePolygon;
}
