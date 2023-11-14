#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "math_utils.hpp"

#include "atomic_single_warp_multiplier.h"
#include "multiple_warp_multiplier.h"
#include "shuffle_single_warp_multiplier.h"

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPITITIONS 10

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