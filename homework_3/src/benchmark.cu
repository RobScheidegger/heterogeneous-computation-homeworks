#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a < b ? b : a)
#define CEIL(x, y) ((x + y - 1) / y)

#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32
#define WARP_SIZE 32

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPITITIONS 10

// math_utils.hpp
template <typename T, typename F = float>
std::pair<F, F> getMeanAndStdDev(const std::vector<T>& v) {
    F sum = std::accumulate(v.begin(), v.end(), 0.0);
    F mean = sum / v.size();

    F sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    F stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stdev);
}

// matrix_vector_multiplier.hpp
class IMatrixVectorMultiplier {
   public:
    typedef std::shared_ptr<IMatrixVectorMultiplier> SharedPtr;

    virtual uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const = 0;

    virtual std::string getName() const = 0;
};

// atomic_single_warp_multiplier.h/cupp
__global__ void saxby_atomic_single(float* A, float* x, int num_rows, int num_cols, float* y) {

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
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const override {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 num_blocks(CEIL(num_rows, WARPS_PER_BLOCK), 1, 1);
        dim3 max_threads_per_block(MAX_THREADS_PER_BLOCK, 1, 1);
        saxby_atomic_single<<<num_blocks, max_threads_per_block>>>(A, x, num_rows, num_cols, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override { return "AtomicSingleWarpMultiplier"; }
};

__global__ void saxby_multiple_warp(float* A, float* x, int num_rows, int num_cols, float* y) {
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
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const override {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 num_blocks(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);
        saxby_multiple_warp<<<num_blocks, max_threads_per_block>>>(A, x, num_rows, num_cols, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override { return "MultipleWarpMultiplier"; }
};

// shuffle_single_warp_multiplier.h/cupp
__global__ void saxby_shuffle_single(float* A, float* x, int num_rows, int num_cols, float* y) {

    int warp_id = threadIdx.x / THREADS_PER_WARP;
    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int matrix_row_idx = warp_id + blockIdx.x * THREADS_PER_WARP;
    int col_start_idx = lane_id * num_cols / THREADS_PER_WARP;
    int col_end_idx = MIN((lane_id + 1) * num_cols / THREADS_PER_WARP, num_cols);

    float sum = 0;
    for (int i = col_start_idx; i < col_end_idx; i++)
        sum += A[matrix_row_idx * num_cols + i] * x[i];
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffff, sum, offset);

    if (lane_id == 0) {
        y[matrix_row_idx] += sum;
    }
}

class ShuffleSingleWarpMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const override {
        auto start = std::chrono::high_resolution_clock::now();

        dim3 num_blocks(CEIL(num_rows, WARPS_PER_BLOCK), 1, 1);
        dim3 max_threads_per_block(MAX_THREADS_PER_BLOCK, 1, 1);
        saxby_shuffle_single<<<num_blocks, max_threads_per_block>>>(A, x, num_rows, num_cols, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override { return "ShuffleSingleWarpMultiplier"; }
};

// wide_matrix_multiplier.h/cupp
__global__ void transpose(float* A, float* A_T, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        A_T[col * num_rows + row] = A[row * num_cols + col];
    }
}

class WideMatrixMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const override {
        float* A_T;

        cudaError_t err = cudaMalloc(&A_T, num_rows * num_cols * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();

        // Each row is a block that contains 1024 threads. This is best since we are only performing trnaspositions on fat matrices.
        dim3 num_blocks_transpose(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block_transpose(1, MAX_THREADS_PER_BLOCK, 1);

        transpose<<<num_blocks_transpose, max_threads_per_block_transpose>>>(A, A_T, num_rows, num_cols);

        dim3 num_blocks(num_rows, CEIL(num_cols, MAX_THREADS_PER_BLOCK), 1);
        dim3 max_threads_per_block(1, MAX_THREADS_PER_BLOCK, 1);
        saxby_shuffle_single<<<num_blocks, max_threads_per_block>>>(A_T, x, num_cols, num_rows, y);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        cudaFree(A_T);
    }

    std::string getName() const override { return "WideMatrixMultiplier"; }
};

void safeCudaMalloc(void** ptr, size_t size) {
    cudaError_t err;
    err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void safeCudaMemcpy(void* dest, void* src, size_t size, cudaMemcpyKind kind) {
    cudaError_t err;
    err = cudaMemcpy(dest, src, size, kind);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

struct BenchmarkConfiguration {
    uint32_t n;
    uint32_t m;
    uint32_t repetitions = BENCHMARK_REPITITIONS;
    IMatrixVectorMultiplier::SharedPtr multiplier;

    std::vector<uint64_t> times;

    BenchmarkConfiguration(uint32_t n, uint32_t m, IMatrixVectorMultiplier::SharedPtr multiplier)
        : n(n), m(m), multiplier(multiplier) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(repetitions) + ',' +
               std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    }
};

int main(int argc, char** argv) {
    std::vector<uint32_t> m_options{10, 100, 1000, 10000, 20000};
    std::vector<float> aspect_ratio_options{0.01, 0.1, 1, 10, 100};
    std::vector<IMatrixVectorMultiplier::SharedPtr> multipliers{
        std::make_shared<AtomicSingleWarpMultiplier>(),
        std::make_shared<MultipleWarpMultiplier>(),
        std::make_shared<ShuffleSingleWarpMultiplier>(),
    };
    std::vector<BenchmarkConfiguration> configurations;

    std::cout << "Creating benchmark configurations..." << std::endl;

    for (auto& m : m_options) {
        for (auto& aspect_ratio : aspect_ratio_options) {
            for (auto& multiplier : multipliers) {
                configurations.emplace_back(BenchmarkConfiguration((uint32_t)m * aspect_ratio, m, multiplier));
            }
        }
    }

    // Atomic vs Shuffle
    // Multiple Warp vs Single Warp
    // Wide vs Multiple Warp (for "fat" matrix, say 10 rows x 10000 columns)

    std::cout << "Found " << configurations.size() << " configurations." << std::endl;

    uint32_t experiment_number = 1;
    for (auto& configuration : configurations) {
        const uint32_t n = configuration.n;
        const uint32_t m = configuration.m;
        const IMatrixVectorMultiplier::SharedPtr multiplier = configuration.multiplier;

        uint32_t garbage = 1;
        for (uint32_t i = 1; i < WARMUP_OPERATIONS; i++) {
            garbage *= i;
        };

        for (uint32_t repetition = 0; repetition < configuration.repetitions; repetition++) {
            float* A_h = (float*)malloc(n * m * sizeof(float));
            float* x_h = (float*)malloc(m * sizeof(float));
            float* y_h = (float*)malloc(n * sizeof(float));

            for (int i = 0; i < n * m; i++) {
                A_h[i] = (float)(rand() % 65536) - 32768;
            }
            for (int i = 0; i < m; i++) {
                x_h[i] = (float)(rand() % 65536) - 32768;
            }
            for (int i = 0; i < n; i++) {
                y_h[i] = (float)(rand() % 65536) - 32768;
            }

            // perform the memcopy to the device
            float *A_d, *x_d, *y_d;

            safeCudaMalloc((void**)&A_d, n * m * sizeof(float));
            safeCudaMalloc((void**)&x_d, m * sizeof(float));
            safeCudaMalloc((void**)&y_d, n * sizeof(float));

            safeCudaMemcpy(A_d, A_h, n * m * sizeof(float), cudaMemcpyHostToDevice);
            safeCudaMemcpy(x_d, x_h, m * sizeof(float), cudaMemcpyHostToDevice);
            safeCudaMemcpy(y_d, y_h, n * sizeof(float), cudaMemcpyHostToDevice);

            // call the kernel
            uint32_t time = multiplier->multiply(A_d, x_d, y_d, n, m);

            configuration.times.push_back(time);
        }

        std::cout << experiment_number << "," << configuration.toCsv() << std::endl;
        experiment_number++;
    }
}