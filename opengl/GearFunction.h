#pragma once

#include <glad/glad.h>

class GearFunction {
public:
    virtual const GLfloat operator()(GLfloat theta) const = 0;
};

class CenterEllipseGearFunction : public GearFunction {
    GLfloat a, b;
public:
    CenterEllipseGearFunction(GLfloat a, GLfloat b) : a(a), b(b) {}

    const GLfloat operator()(GLfloat theta) const override;
};