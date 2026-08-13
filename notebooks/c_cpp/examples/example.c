#include <stdio.h>

#define MACRO_EXAMPLE(x) ((x) + (x))

typedef int TypeExample;

struct StructExample {
    int x;
    int y;
};

static int global_variable_example = 42;

int function_example(int a, int b) {
    return a + b;
}

static int static_function_example(const struct StructExample *se) {
    return MACRO_EXAMPLE(se->x) + se->y;
}

int main(void) {
    int result = function_example(2, 3);

    if (result > 4) {
        printf("%d\n", result);
    }

    return 0;
}