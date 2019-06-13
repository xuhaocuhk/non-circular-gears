#include <utility>

#include "GearDisplay.h"
#include "utility.h"

GearDisplay::GearDisplay(size_t sampleCount, GearFunction *function, GLfloat centerX, GLfloat centerY,
                         std::shared_ptr<Shader> shader) : rotation(0), centerX(centerX), centerY(centerY),
                                                           shader(std::move(shader)) {
    //prepare vertices and indicies
    vertices.reserve(sampleCount * 2 + 2);
    vertices.emplace_back(0);
    vertices.emplace_back(0);
    const auto unitAngle = 2 * util::PI / sampleCount;
    for (int i = 0; i < sampleCount; i++) {
        const auto angle = unitAngle * i;
        const auto r = (*function)(angle);
        vertices.emplace_back(r * cos(angle));
        vertices.emplace_back(r * sin(angle));
    }
    indices.reserve(sampleCount * 3);
    for (int i = 1; i <= sampleCount; i++) {
        indices.emplace_back(0);
        indices.emplace_back(i);
        indices.emplace_back(i + 1);
    }
    indices.emplace_back(0);
    indices.emplace_back(sampleCount);
    indices.emplace_back(1);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(decltype(vertices)::value_type), vertices.data(),
                 GL_STATIC_DRAW);
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(decltype(indices)::value_type), indices.data(),
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(0);
}

void GearDisplay::draw(GLfloat xMin, GLfloat xMax, GLfloat yMin, GLfloat yMax, bool fill) {
    shader->use();
    shader->set("rotation", rotation);
    shader->set("translation", -centerX, -centerY);
    shader->set("xMin", xMin);
    shader->set("xMax", xMax);
    shader->set("yMin", yMin);
    shader->set("yMax", yMax);

    if (fill) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size() / 3, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

GearDisplay::~GearDisplay() {
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
}
