#version 330 core
layout (location = 0) in vec2 vPos;
uniform vec2 translation;
uniform float rotation;
uniform float xMin, xMax, yMin, yMax;

void main() {
    vPos = vec2(vPos.x*cos(rotation)-vPos.y*sin(rotation), vPos.y*sin(rotation)+vPos.x*cos(rotation));
    vPos = vPos+translation;
    vPos = vec2((vPos.x - xMin)/(xMax-xMin), (vPos.y - yMin)/(yMax-yMin));
    gl_Position = vec4(vPos, 1.0, 1.0);
}
