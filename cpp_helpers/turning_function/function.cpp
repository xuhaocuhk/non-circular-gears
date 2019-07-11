#include "function.h"
#include <set>

using namespace std;

Function::Function(const EdgePolygon &polygon) {
    //TODO
}

double Function::at(double x) const {
    assert(x >= rangeStart - EPS && x <= rangeEnd + EPS);
    assert(isDataLegal());
    x += EPS; //avoid finding error
    const auto &index = lower_bound(points.begin(), points.end(), x) - points.begin();
    return values[index];
}

inline double squareDifference(const double &x, const double &y) {
    double d = x - y;
    return d * d;
}

double Function::distanceTo(const Function &rhs) const {
    auto selfIter = points.begin();
    auto otherIter = rhs.points.begin();
    double sum = 0;
    double lastPoint = 0;
    while (selfIter != points.end() && otherIter != rhs.points.end()) {
        double current;
        if (*selfIter < *otherIter) current = *selfIter, ++selfIter;
        else current = *otherIter, ++otherIter;
        sum += (current - lastPoint) * squareDifference(at(current), rhs.at(current));
        lastPoint = current;
    }
    while (selfIter != points.end()) {
        sum += (*selfIter - lastPoint) * squareDifference(at(*selfIter), rhs.at(*selfIter));
        lastPoint = *selfIter;
        ++selfIter;
    }
    while (otherIter != points.end()) {
        sum += (*otherIter - lastPoint) * squareDifference(at(*otherIter), rhs.at(*otherIter));
        lastPoint = *otherIter;
        ++otherIter;
    }
    return sum;
}
