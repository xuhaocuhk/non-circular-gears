#include "function.h"
#include <set>

Function::Function(const EdgePolygon &polygon) {
    //TODO
}

double Function::at(double x) const {
    assert(x >= rangeStart - EPS && x <= rangeEnd + EPS);
    assert(!data.empty());
    x += EPS; //avoid map error
    return data.lower_bound(x)->second;
}

inline double squareDifference(const double &x, const double &y) {
    double d = x - y;
    return d * d;
}

double Function::distanceTo(const Function &rhs) const {
    //TODO: maybe we can use a list?
    std::set<double> xPoints;
    double sum = 0;
    assert(fabs(rhs.rangeStart - rangeStart) <= EPS && fabs(rhs.rangeEnd - rangeEnd) <= EPS);

    for (const auto &datum:data) xPoints.insert(datum.first);
    for (const auto &datum:rhs.data) xPoints.insert(datum.first);
    for (auto iter = xPoints.begin(); iter != xPoints.end();) {
        double x = *iter;
        double nextX = (++iter) == xPoints.end() ? rangeEnd : *iter;
        sum += squareDifference(rhs.at(x), this->at(x)) * (nextX - x);
    }
    return sum;
}
