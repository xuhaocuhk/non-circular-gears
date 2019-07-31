#include <cstdio>

int main(int argc, char *argv[]) {
    int len;

    if (argc > 1) {
        //read file from given argument
        freopen(argv[1], "rb", stdin);
    } else {
        //read from stdin
        freopen(nullptr, "rb", stdin);
    }

    fread(&len, sizeof(len), 1, stdin);
    printf("%d\n", len);
    for (int i = 0; i < len; i++) {
        double x, y;
        fread(&x, sizeof(x), 1, stdin);
        fread(&y, sizeof(y), 1, stdin);
        printf("%.6lf %.6lf\n", x, y);
    }

    return 0;
}