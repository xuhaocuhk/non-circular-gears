#pragma once

#include <glad/glad.h>

class GearFunction {
public:
    virtual const GLfloat operator()(GLfloat theta) const = 0;
};

class CircleGearFunction : public GearFunction {
private:
    GLfloat radius;
public:
    CircleGearFunction(GLfloat radius) : radius(radius) {}

    const GLfloat operator()(GLfloat theta) const override { return radius; }
};

class CenterEllipseGearFunction : public GearFunction {
    GLfloat a, b;
public:
    CenterEllipseGearFunction(GLfloat a, GLfloat b) : a(a), b(b) {}

    const GLfloat operator()(GLfloat theta) const override;
};