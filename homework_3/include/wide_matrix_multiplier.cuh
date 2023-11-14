#include <cuda.h>

#include "./matrix_vector_multiplier.hpp"
#include "./multiple_warp_multiplier.cuh"

__global__ transpose(float* A, float* A_T, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        A_T[col * num_rows + row] = A[row * num_cols + col];
    }
}

class WideMatrixMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) override {
        float* A_T;

        A_T = cudaMalloc(num_rows * num_cols * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();

        // Each row is a block that contains 1024 threads. This is best since we are only performing trnaspositions on fat matrices.
        dim3 num_blocks(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);

        transpose<<<num_blocks, max_threads_per_block>>>(A, A_T, num_rows, num_cols);

        dim3 num_blocks(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);
        saxby<<<num_blocks, max_threads_per_block>>>(A_T, x, num_cols, num_rows, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        free(A_T);
    }
}