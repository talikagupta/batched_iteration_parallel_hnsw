#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(status) \
    if (status != 0) { \
        std::cerr << "CUDA Error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

void batched_L2(
    cublasHandle_t handle,
    const float* x1,
    const float* x2,
    float* output,
    int B, int R, int M
);

// KERNEL declarations
__global__ void compute_x1_norm(const float* x1, float* x1_norm, int B, int M);
__global__ void compute_x2_norm(const float* x2, float* x2_norm, int B, int R, int M);
__global__ void add_norms(float* output, const float* x1_norm, const float* x2_norm, int B, int R);

// CUDA kernel to compute x1 norm
__global__ void compute_x1_norm(const float* x1, float* x1_norm, int B, int M) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("compute_x1_norm kernel started\n");
    }
    if (b >= B) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        float val = x1[b * M + m];
        sum += val * val;
    }
    x1_norm[b] = sum;
}

// CUDA kernel to compute x2 norm
__global__ void compute_x2_norm(const float* x2, float* x2_norm, int B, int R, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= B * R) return;

    int b = idx / R;
    int r = idx % R;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        float val = x2[b * R * M + r * M + m];
        sum += val * val;
    }
    x2_norm[b * R + r] = sum;
}

// CUDA kernel to add norms
__global__ void add_norms(float* output, const float* x1_norm, const float* x2_norm, int B, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * R) return;

    int b = idx / R;
    int r = idx % R;

    int offset = b * R + r;
    output[offset] += x1_norm[b] + x2_norm[offset];
    output[offset] = fmaxf(output[offset], 1e-30f);
}

void batched_L2(cublasHandle_t handle,
    const float* x1,  // (B, 1, M)
    const float* x2,  // (B, R, M)
    float* output,    // (B, 1, R)
    int B, int R, int M
) {
    const float alpha = -2.0f;
    const float beta = 0.0f;

    // x1 * x2^T: we want (B, 1, R)

    size_t strideA = M;      // x1: (1, M)
    size_t strideB = R * M;  // x2: (R, M)
    size_t strideC = R;      // output: (1, R)

    // Input matrices stored as row major
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        R, 1, M,
        &alpha,
        x2, M, strideB,
        x1, M, strideA,
        &beta,
        output, M, strideC,
        B
    ));

    // Now add x1_norm and x2_norm

    // --- Compute norms on the device
    // We need:
    // - x1_norm: (B, 1)
    // - x2_norm: (B, R)    

    // Allocate memory for norms
    // Compute x1_norm
    // x1_norm[b] = sum(x1[b, 0, m]^2) over m
    // Similarly for x2_norm
    float* x1_norm;
    float* x2_norm;
    
    cudaError_t err = cudaMalloc(&x1_norm, B * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc x1_norm failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&x2_norm, B * R * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc x2_norm failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        return;
    }
    
    // Launch kernels with proper error checking
    dim3 threads(256);  // will need more threads if dim > 256
    dim3 blocks_x1((B + 255) / 256);
    
    // Launch first kernel
    compute_x1_norm<<<blocks_x1, threads>>>(x1, x1_norm, B, M);
    
    // Check for kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_x1_norm launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }

    // Wait for kernel to finish and check for execution error
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("compute_x1_norm execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }
    
    // Launch second kernel
    dim3 blocks_x2((B * R + 255) / 256);
    compute_x2_norm<<<blocks_x2, threads>>>(x2, x2_norm, B, R, M);
    
    // Check for kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_x2_norm launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }

    
    // Wait for kernel to finish and check for execution error
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("compute_x2_norm execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }
    
    // Final add: output[b, 0, r] += x1_norm[b] + x2_norm[b, r]
    dim3 blocks_output((B * R + 255) / 256);
    add_norms<<<blocks_output, threads>>>(output, x1_norm, x2_norm, B, R);
    
    // Check for kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("add_norms launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }
    
    // Wait for kernel to finish and check for execution error
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("add_norms execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(x1_norm);
        cudaFree(x2_norm);
        return;
    }

    // Print final output
    // {
    //     std::vector<float> host_output(B * R);
    //     cudaMemcpy(host_output.data(), output, B * R * sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("Final output:\n");
    //     for (int b = 0; b < B; ++b) {
    //         printf("  batch %d:", b);
    //         for (int r = 0; r < R; ++r)
    //             printf(" %f", host_output[b * R + r]);
    //         printf("\n");
    //     }
    // }
    
    // Free norm buffers
    cudaFree(x1_norm);
    cudaFree(x2_norm);
}


