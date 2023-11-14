#pragma once

#include "./matrix_vector_multiplier.hpp"

__global__ void saxby(float* A, float* x, int num_rows, int num_cols, float* y) {

    int warp_id = threadIdx.x / THREADS_PER_WARP;
    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int matrix_row_idx = warp_id + blockIdx.x * THREADS_PER_WARP;
    int col_start_idx = lane_id * num_cols / THREADS_PER_WARP;
    int col_end_idx = MIN((lane_id + 1) * num_cols / THREADS_PER_WARP, num_cols);

    float sum = 0;
    for (int i = col_start_idx; i < col_end_idx; i++)
        sum += A[matrix_row_idx * num_cols + i] * x[i];
    __syncthreads();

    atomicAdd(&y[matrix_row_idx], sum);
}

class AtomicSingleWarpMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) override {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 num_blocks(CEIL(num_rows, WARPS_PER_BLOCK), 1, 1);
        dim3 max_threads_per_block(MAX_THREADS_PER_BLOCK, 1, 1);
        saxby<<<num_blocks, max_threads_per_block>>>(A, x, num_rows, num_cols, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
}