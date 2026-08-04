#include <stdio.h>

#define SQUARE(x) ((x) * (x))

typedef int MyInt;

struct Point {
    int x;
    int y;
};

static int global_counter = 42;

int add(int a, int b) {
    return a + b;
}

static int helper(const struct Point *p) {
    return SQUARE(p->x) + p->y;
}

int main(void) {
    int result = add(2, 3);

    if (result > 4) {
        printf("%d\n", result);
    }

    return 0;
}