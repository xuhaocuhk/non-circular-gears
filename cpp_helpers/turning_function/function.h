#pragma once

#include "geometry.h"
#include <algorithm>

/**
 * Staircase Functions
 */
struct Function {
    std::vector<double> points, values;

    inline explicit Function(double rangeStart = 0, double rangeEnd = 0) : rangeStart(rangeStart), rangeEnd(rangeEnd) {}

    Function(const EdgePolygon &polygon); //generate turning function from polygon NOLINT(google-explicit-constructor)

    inline double getRangeStart() const { return rangeStart; }

    inline double getRangeEnd() const { return rangeEnd; }

    double at(double x) const;

    double distanceTo(const Function &rhs) const;

private:
    double rangeStart, rangeEnd;

    bool isDataLegal() const {
        return (!points.empty()) &&
               fabs(points[0] - rangeStart) <= EPS &&
               fabs(points[points.size() - 1] - rangeEnd) <= EPS &&
               points.size() == values.size() &&
               std::is_sorted(points.begin(), points.end());
    }
};