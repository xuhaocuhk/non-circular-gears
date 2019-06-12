//
// Created by kevin on 2019-06-12.
//

#include "GearFunction.h"
#include "utility.h"

const GLfloat CenterEllipseGearFunction::operator()(GLfloat theta) const {
    const auto &x = b * cos(theta);
    const auto &y = a * sin(theta);
    return a * b / sqrt(x * x + y * y);
}
