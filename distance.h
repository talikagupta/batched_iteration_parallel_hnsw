#pragma once

#include <cublas_v2.h>

void batched_L2(
    cublasHandle_t handle,
    const float* x1,
    const float* x2,
    float* output,
    int B, int R, int M
);

