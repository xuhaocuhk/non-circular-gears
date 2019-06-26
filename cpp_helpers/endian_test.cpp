#include <iostream>
#include <cstdio>

int main() {
    //test data
    double d;
    int i;
    char c;

    //start binary mode
    freopen(nullptr, "rb", stdin);

    fread(&d, sizeof(d), 1, stdin);
    fread(&i, sizeof(i), 1, stdin);
    fread(&c, sizeof(c), 1, stdin);

    printf("%03.5lf %05d %c", d, i, c);

    return 0;
}