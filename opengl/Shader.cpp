#include "Shader.h"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

Shader::Shader(const char *vertexShaderPath, const char *fragmentShaderPath) {
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    vShaderFile.open(vertexShaderPath);
    fShaderFile.open(fragmentShaderPath);
    std::stringstream vShaderStream, fShaderStream;
    vShaderStream << vShaderFile.rdbuf();
    fShaderStream << fShaderFile.rdbuf();
    vShaderFile.close();
    fShaderFile.close();
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();

    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();
    GLuint vertex, fragment;
    int success;
    char infoLog[512];
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex, 512, nullptr, infoLog);
        std::cerr << "Unable to compile vertex shader:\n" << infoLog << std::endl;
    }
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment, 512, nullptr, infoLog);
        std::cerr << "Unable to compile fragment shader:\n" << infoLog << std::endl;
    }

    this->id = glCreateProgram();
    glAttachShader(id, vertex);
    glAttachShader(id, fragment);
    glLinkProgram(id);
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(id, 512, nullptr, infoLog);
        std::cout << "failed to link program: \n" << infoLog << std::endl;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() const {
    glUseProgram(this->id);
}

void Shader::set(const char *variable, GLboolean value) const {
    glUniform1i(glGetUniformLocation(id, variable), static_cast<int>(value));
}

void Shader::set(const char *variable, GLint value) const {
    glUniform1i(glGetUniformLocation(id, variable), value);
}

void Shader::set(const char *variable, GLfloat value) const {
    glUniform1f(glGetUniformLocation(id, variable), value);
}

void Shader::set(const char *variable, GLfloat x, GLfloat y) const {
    glUniform2f(glGetUniformLocation(id, variable), x, y);
}

void Shader::set(const char *variable, GLfloat x, GLfloat y, GLfloat z) const {
    glUniform3f(glGetUniformLocation(id, variable), x, y, z);
}

Shader::~Shader() {
    glDeleteProgram(this->id);
}
