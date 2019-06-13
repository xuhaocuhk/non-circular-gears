#pragma once

#include <glad/glad.h>
#include <memory>
#include <vector>
#include <cstddef>
#include "GearFunction.h"
#include "Shader.h"

class GearDisplay {
private:
    std::vector<GLfloat> vertices;
    std::vector<GLuint> indices;
    GLfloat rotation;
    GLfloat centerX, centerY;
    std::shared_ptr<Shader> shader;

    GLuint VAO, VBO, EBO;
public:
    GearDisplay(size_t sampleCount, GearFunction *function, GLfloat centerX, GLfloat centerY,
                std::shared_ptr<Shader> shader);

    const std::vector<GLfloat> &getVertices() const { return vertices; }

    void rotate(GLfloat theta) { rotation += theta; }

    void setRotationAngle(GLfloat angle) { rotation = angle; }

    void draw(GLfloat xMin, GLfloat xMax, GLfloat yMin, GLfloat yMax, bool fill = false);

    ~GearDisplay();
};
