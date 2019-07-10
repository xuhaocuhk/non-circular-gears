#include <cstdio>
#include <vector>
#include <cassert>

using std::vector;

struct Point {
    double x, y;
};
typedef vector<Point> Polygon;

const double EPS = 1e-5;

void readPolygon(Polygon &polygon);
void printPolygon(const char *name, const Polygon &polygon);

int main(int argc, char **argv) {
    int targetSize, gearSize;
    vector<Point> targetPolygon;
    vector<Point> gearPolygon;

    freopen(nullptr, "rb", stdin);

    readPolygon(targetPolygon);
    readPolygon(gearPolygon);

    printPolygon("target polygon", targetPolygon);
    printPolygon("gear polygon", gearPolygon);

    return 0;
}

void readPolygon(Polygon &polygon){
    int size;
    fread(&size, sizeof(int), 1, stdin);
    polygon.resize(size);
    assert(sizeof(Point) == 2 * sizeof(double));
    fread(polygon.data(), sizeof(double), size * 2, stdin);
}

void printPolygon(const char *name, const Polygon &polygon){
    printf("%s:\n",name);
    for(const auto &point : polygon)
        printf("(%.4lf,%.4lf)\n", point.x, point.y);
}