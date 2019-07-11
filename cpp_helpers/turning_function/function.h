#pragma once

#include <map>
#include "geometry.h"

/**
 * Staircase Functions
 */
struct Function {
    std::map<double, double> data;

    inline explicit Function(double rangeStart = 0, double rangeEnd = 0) : rangeStart(rangeStart), rangeEnd(rangeEnd) {}

    Function(const EdgePolygon &polygon); //generate turning function from polygon

    inline double getRangeStart() const { return rangeStart; }

    inline double getRangeEnd() const { return rangeEnd; }

    double at(double x) const;

    double distanceTo(const Function &rhs) const;

private:
    double rangeStart, rangeEnd;
};