#version 330 core
layout (location = 0) in vec2 vPos;
uniform vec2 translation;
uniform float rotation;
uniform float xMin, xMax, yMin, yMax;

void main() {
    vec2 pos;
    pos = vec2(vPos.x*cos(rotation)-vPos.y*sin(rotation), vPos.x*sin(rotation)+vPos.y*cos(rotation));
    pos = pos+translation;
    pos = vec2((pos.x - xMin)/(xMax-xMin)*2-1.0f, (pos.y - yMin)/(yMax-yMin)*2-1.0f);
    gl_Position = vec4(pos, 0.0, 1.0);
}
