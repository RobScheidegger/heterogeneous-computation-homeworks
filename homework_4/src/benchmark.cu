#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <sys/types.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
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

#define WARMUP_OPERATIONS 10
#define BENCHMARK_REPITITIONS 10

template <typename T, typename F = float>
std::pair<F, F> getMeanAndStdDev(const std::vector<T>& v) {
    F sum = std::accumulate(v.begin(), v.end(), 0.0);
    F mean = sum / v.size();

    F sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    F stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stdev);
}

class IMatrixVectorMultiplier {
   public:
    typedef std::shared_ptr<IMatrixVectorMultiplier> SharedPtr;

    virtual uint32_t multiply(float* const A, float* const x, float* y, float* r, int num_rows, int num_cols,
                              int num_streams) const = 0;

    virtual std::string getName() const = 0;
};

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

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffff, sum, offset);

    if (lane_id == 0) {
        y[matrix_row_idx] += sum;
    }
}

class MatrixVectorStreamMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, float* r, int num_rows, int num_cols,
                      int num_streams) const override {

        auto start = std::chrono::high_resolution_clock::now();

        cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        int rows_per_stream = CEIL(num_rows, num_streams);
        for (int i = 0; i < num_streams; i++) {
            int start_row = i * rows_per_stream;
            int end_row = MIN((i + 1) * rows_per_stream, num_rows);
            int num_rows_in_stream = end_row - start_row;
            int num_bytes = num_rows_in_stream * num_cols * sizeof(float);

            float* A_d;
            cudaMallocAsync(&A_d, num_bytes, streams[i]);
            cudaMemcpyAsync(A_d, A + start_row * num_cols, num_bytes, cudaMemcpyHostToDevice, streams[i]);

            dim3 num_blocks(CEIL(num_rows_in_stream, WARPS_PER_BLOCK), 1, 1);
            dim3 max_threads_per_block(MAX_THREADS_PER_BLOCK, 1, 1);
            saxby<<<num_blocks, max_threads_per_block, 0, streams[i]>>>(A_d, x, num_rows_in_stream, num_cols,
                                                                        y + start_row);
            cudaMemcpyAsync(r + start_row, y + start_row, num_rows_in_stream * sizeof(float), cudaMemcpyDeviceToHost,
                            streams[i]);
            cudaFreeAsync(A_d, streams[i]);
        }

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override { return "MatrixVectorStreamMultiplier"; }
};

struct BenchmarkConfiguration {
    uint32_t n;
    uint32_t m;
    uint32_t k;
    uint32_t repetitions = BENCHMARK_REPITITIONS;
    IMatrixVectorMultiplier::SharedPtr multiplier;

    std::vector<uint64_t> times;

    BenchmarkConfiguration(uint32_t n, uint32_t m, uint32_t k, IMatrixVectorMultiplier::SharedPtr multiplier)
        : n(n), m(m), k(k), multiplier(multiplier) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(k) + ',' + std::to_string(repetitions) + ',' +
               multiplier->getName() + ',' + std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    }
};

void safeCudaMallocHost(void** ptr, size_t size) {
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
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

void safeCudaMemcpy(void* dest, void* src, size_t size, cudaMemcpyKind kind) {
    cudaError_t err;
    err = cudaMemcpy(dest, src, size, kind);
    if (err != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void generateParameters(float **A, float **x_h, float **y_h, float **x_d, float **y_d, int n, int k, int m) {
    *x_h = (float*)malloc(k * sizeof(float));
    safeCudaMallocHost((void**)A, n * k * sizeof(float));
    safeCudaMallocHost((void**)y_h, n * sizeof(float));

    for (int i = 0; i < n * k; i++) {
        (*A)[i] = 1;
    }

    for (int i = 0; i < k; i++) {
        (*x_h)[i] = 1;
    }

    for (int i = 0; i < n; i++) {
        (*y_h)[i] = 0;
    }

    safeCudaMalloc((void**)x_d, k * sizeof(float));
    safeCudaMalloc((void**)y_d, n * sizeof(float));
    safeCudaMemcpy(*x_d, *x_h, k * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(*y_d, *y_h, n * sizeof(float), cudaMemcpyHostToDevice);
}

void freeParameters(float* A, float *x_h, float *y_h, float* x_d, float *y_d) {
    cudaFreeHost(A);
    cudaFreeHost(y_h);
    cudaFree(x_d);
    cudaFree(y_d);

    free(x_h);
}

int main() {
    std::vector<uint32_t> n_options = {1000, 1200, 1400, 1600, 1800, 2000};
    std::vector<uint32_t> k_options = {1000, 1200, 1400, 1600, 1800, 2000};
    std::vector<uint32_t> m_options = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<IMatrixVectorMultiplier::SharedPtr> multipliers = {std::make_shared<MatrixVectorStreamMultiplier>()};

    std::vector<BenchmarkConfiguration> configurations;
    std::cout << "Generating configurations..." << std::endl;

    for (auto n : n_options) {
        for (auto k : k_options) {
            for (auto m : m_options) {
                for (auto multiplier : multipliers) {
                    configurations.emplace_back(n, m, k, multiplier);
                }
            }
        }
    }

    std::cout << "Generated " << configurations.size() << " configurations" << std::endl;
    std::cout << "n,m,k,reps,multiplier,time_us,stddev_us";
    uint32_t experiment_number = 1;
    for (auto& configuration : configurations) {
        const uint32_t n = configuration.n;
        const uint32_t k = configuration.k;
        const uint32_t m = configuration.m;
        const IMatrixVectorMultiplier::SharedPtr multiplier = configuration.multiplier;

        float *A, *x_h, *y_h, *x_d, *y_d;
        generateParameters(&A, &x_h, &y_h, &x_d, &y_d, n, k, m);
        for (uint32_t i = 1; i < WARMUP_OPERATIONS; i++) {
            multiplier->multiply(A, x_d, y_d, y_h, n, k, m);
        };


        for (uint32_t repetition = 0; repetition < configuration.repetitions; repetition++) {
            float *A, *x_h, *y_h, *x_d, *y_d;
            generateParameters(&A, &x_h, &y_h, &x_d, &y_d, n, k, m);
            
            uint32_t time = multiplier->multiply(A, x_d, y_d, y_h, n, k, m);
            configuration.times.push_back(time);

            freeParameters(A, x_h, y_h, x_d, y_d);
        }

        std::cout << experiment_number << "," << configuration.toCsv() << std::endl;
        experiment_number++;
    }
}