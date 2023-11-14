#pragma once

#include "./matrix_vector_multiplier.hpp"

__global__ void saxby(float* A, float* x, int num_rows, int num_cols, float* y) {
    int matrix_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute the dot product between A[matrix_row_idx : matrix_row_idx + num_cols] and x
    // (recall that A is a 1D flattened vector representing a matrix)

    // telling us which part of the column in A we're in for the given matrix_row_id
    int id = threadIdx.y + blockIdx.y * blockDim.y;

    int nwarps = blockDim.y / THREADS_PER_WARP;
    int warp_id = threadIdx.y / THREADS_PER_WARP;
    int lane_id = threadIdx.y % THREADS_PER_WARP;

    // Shared memory is local to the thread block, so we don't need to worry about other blocks overwritting this
    __shared__ float shared_dot_product[MAX_THREADS_PER_BLOCK / THREADS_PER_WARP];

    float sum = 0.0f;
    if (id < num_cols) {
        sum = A[matrix_row_idx * num_cols + id] * x[id];
    }
    __syncthreads();

    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffff, sum, offset);

    if (lane_id == 0)
        shared_dot_product[warp_id] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
        float sum = 0;
        for (int i = 0; i < nwarps; i++) {
            sum += shared_dot_product[i];
        }
        atomicAdd(&y[matrix_row_idx], sum);
    }
}

class MultipleWarpMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) override {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 num_blocks(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);
        saxby<<<num_blocks, max_threads_per_block>>>(A, x, num_rows, num_cols, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
}