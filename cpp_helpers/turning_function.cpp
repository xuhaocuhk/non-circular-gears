#include <cstdio>
#include <vector>
#include <cassert>

using std::vector;

struct Point {
    double x, y;
};
vector<Point> targetPolygon;
vector<Point> gearPolygon;

int main() {
    int targetSize, gearSize;

    freopen(nullptr, "rb", stdin);

    fread(&targetSize, sizeof(targetSize), 1, stdin);
    targetPolygon.resize(targetSize);
    assert(sizeof(Point) == 2 * sizeof(double));
    fread(targetPolygon.data(), sizeof(double), targetSize * 2, stdin);

    fread(&gearSize, sizeof(gearSize), 1, stdin);
    gearPolygon.resize(gearSize);
    fread(gearPolygon.data(), sizeof(double), targetSize * 2, stdin);

    printf("target polygon:\n");
    for(const auto &point : targetPolygon)
        printf("(%.4lf,%.4lf)\n", point.x, point.y);

    printf("gear polygon:\n");
    for(const auto &point : gearPolygon)
        printf("(%.4lf,%.4lf)\n", point.x, point.y);

    return 0;
}