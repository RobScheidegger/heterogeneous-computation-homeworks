#include <cuda.h>
#include <cstdio>

#define N 5
#define M 10
#define MAX_THREADS_PER_BLOCK 1024

#define CEIL(x, y) ((x + y - 1) / y)

#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32

void safeCudaMemcpy(void* dest, void* src, size_t size, cudaMemcpyKind kind) {
    cudaError_t err;
    err = cudaMemcpy(dest, src, size, kind);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void safeCudaMalloc(void** ptr, size_t size) {
    cudaError_t err;
    err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void transpose(float* A, float* A_T, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        A_T[col * num_rows + row] = A[row * num_cols + col];
    }
}

int main() {
    float* A_h = (float*)malloc(N * M * sizeof(float));
    float* A_T_h = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < N * M; i++) {
        A_h[i] = i;
    }

    for (int i = 0; i < N * M; i++) {
        A_T_h[i] = 0.0f;
    }

    for (int i = 0; i < N * M; i++) {
        printf("%f ", A_h[i]);
        if (i % M == M - 1) {
            printf("\n");
        }
    }

    // perform the memcopy to the device
    float *A_d, *A_T_d;

    safeCudaMalloc((void**)&A_d, N * M * sizeof(float));
    safeCudaMalloc((void**)&A_T_d, N * M * sizeof(float));

    safeCudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(A_T_d, A_T_h, N * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 num_blocks(N, CEIL(M, MAX_THREADS_PER_BLOCK), 1);
    dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);

    transpose<<<num_blocks, max_threads_per_block>>>(A_d, A_T_d, N, M);

    safeCudaMemcpy(A_h, A_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    safeCudaMemcpy(A_T_h, A_T_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Transpose:\n");

    for (int i = 0; i < N * M; i++) {
        printf("%f ", A_T_h[i]);
        if (i % N == N - 1) {
            printf("\n");
        }
    }

    free(A_h);
    free(A_T_h);
}
