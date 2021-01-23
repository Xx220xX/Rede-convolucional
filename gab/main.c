#include <stdio.h>
#include <time.h>
#include "gabriela_gpu.h"


int main() {
    testXor("gpu_kernels.cl");
    return 0;
}
