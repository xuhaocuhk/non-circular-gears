#pragma once

#include <glad/glad.h>

class Shader {
public:
    Shader(const char *vertexShaderPath, const char *fragmentShaderPath);

    void use() const;

    void set(const char *variable, GLboolean value) const;

    void set(const char *variable, GLint value) const;

    void set(const char *variable, GLfloat value) const;

    void set(const char *variable, GLfloat x, GLfloat y) const;

    void set(const char *variable, GLfloat x, GLfloat y, GLfloat z) const;

    ~Shader();

private:
    GLuint id;
};

