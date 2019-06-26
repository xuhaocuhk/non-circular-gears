#include <iostream>
#include <cstdio>

int main() {
    char c;

    while ((c=getchar())!=EOF) {
        putchar(c);
        fflush(stdout);
    }

    return 0;
}