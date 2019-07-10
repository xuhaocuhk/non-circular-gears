#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <type_traits>

struct Point {
    double x, y;

    Point() = default;

    Point(double x, double y) : x(x), y(y) {}
};

struct Vector : public Point {
    Vector() = default;

    Vector(double x, double y) : Point(x, y) {}

    inline double length() const { return sqrt(x * x + y * y); }
};

typedef std::vector<Point> PointPolygon;
typedef std::vector<Vector> EdgePolygon;

const double EPS = 1e-5;

inline const Vector operator-(const Point &lhs, const Point &rhs) {
    return Vector{rhs.x - lhs.x, rhs.y - lhs.y};
}

/**
 * take modulo in the math way
 * @param x the divided
 * @param m the divisor
 * @return integer in [0, m-1]
 */
inline int mathMod(const int &x, const int &m) {
    auto r = x % m;
    return r < 0 ? r + m : r;
}

/**
 * use as a cyclic index accessor for Container (polygons)
 * @tparam Container vector<Point/Vector>
 * @param polygon the container needs to access
 * @param index index, will access mathMod(index, size)
 * @return const Container::value_type&
 */
template<typename Container>
inline auto at(const Container &polygon, int index) -> decltype(polygon[0]) {
    index = mathMod(index, polygon.size());
    return polygon[index];
}

inline double dot(const Vector &p1, const Vector &p2) { return p1.x * p2.x + p1.y * p2.y; }

inline double cross(const Vector &p1, const Vector &p2) { return p1.x * p2.y - p1.y * p2.x; }

/**
 * check if the edge polygon forms a closed loop
 */
inline bool checkEdgePolygon(const EdgePolygon &polygon) {
    double xSum, ySum;
    xSum = ySum = 0;
    for (const auto &edge: polygon) {
        xSum += edge.x;
        ySum += edge.y;
    }
    return fabs(xSum) < EPS && fabs(ySum) < EPS;
}

EdgePolygon toEdgePolygon(const PointPolygon &polygon);

double perimeter(const EdgePolygon &polygon);

/**
 * calculate the turning angle between the edge and next edge
 * @return angle between edges index and index + 1
 */
double turningAngle(const EdgePolygon &polygon, int index);
