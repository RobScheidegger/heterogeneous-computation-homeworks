
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "allocators.hpp"
#include "math_utils.hpp"
#include "multipliers.hpp"

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPETITIONS 10

struct BenchmarkConfiguration {
    // Configuration fields
    uint32_t n;
    uint32_t m;
    uint32_t num_threads;
    IMatrixVectorAllocator::SharedPtr allocator;
    IMultiplier::SharedPtr multiplier;
    uint32_t repetitions = BENCHMARK_REPETITIONS;

    // Result fields
    std::vector<uint64_t> times;

    BenchmarkConfiguration(const uint32_t n, const uint32_t m, const uint32_t num_threads,
                           IMatrixVectorAllocator::SharedPtr allocator, IMultiplier::SharedPtr multiplier)
        : n(n), m(m), num_threads(num_threads), allocator(allocator), multiplier(multiplier) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(num_threads) + ',' +
               allocator->getName() + ',' + multiplier->getName() + ',' + std::to_string(repetitions) + ',' +
               std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    };
};

const std::vector<uint32_t> n_options{1, 10, 100, 1000, 10000, 100000};
const std::vector<uint32_t> m_options{1, 10, 100, 1000, 10000, 100000};
const std::vector<uint32_t> num_threads_options{1, 2, 4, 8, 16, 32, 64};

int main(int argc, char** argv) {
    std::vector<IMatrixVectorAllocator::SharedPtr> allocators{
        std::make_shared<DisjointMemoryAllocator>(), std::make_shared<DisjointRowMemoryAllocator>(),
        std::make_shared<ContiguousMemoryAllocator>(), std::make_shared<MmapMemoryAllocator>()};

    std::vector<IMultiplier::SharedPtr> multipliers{std::make_shared<RowColumnMultiplier>(),
                                                    std::make_shared<ColumnRowMultiplier>()};

    std::vector<BenchmarkConfiguration> configurations;

    std::cout << "Creating benchmark configurations..." << std::endl;

    if (argc == 2 && std::string(argv[1]) == "--simple") {
        std::cout << "Running simple mode..." << std::endl;
        configurations.emplace_back(100000, 100000, 16, std::make_shared<MmapMemoryAllocator>(),
                                    std::make_shared<RowColumnMultiplier>());
    } else {
        // Make all of the required configurations
        for (auto& n : n_options) {
            for (auto& m : m_options) {
                for (auto& num_threads : num_threads_options) {
                    for (auto& allocator : allocators) {
                        for (auto& multiplier : multipliers) {
                            configurations.emplace_back(n, m, num_threads, allocator, multiplier);
                        }
                    }
                }
            }
        }
    }

    std::cout << "Found " << configurations.size() << " benchmark configurations." << std::endl;

    // Run an experiment for each of the configurations
    for (auto& configuration : configurations) {
        const uint32_t n = configuration.n;
        const uint32_t m = configuration.m;

        // Warmup with some garbage computation
        uint32_t q = 1;
        for (uint32_t i = 1; i < WARMUP_OPERATIONS; i++) {
            q *= i;
        };

        for (uint32_t iteration = 0; iteration < configuration.repetitions; iteration++) {
            float** matrix;
            float* vector;
            float* output;

            if (!configuration.allocator->allocate(n, m, matrix, vector, output))
                continue;

            const uint64_t time =
                configuration.multiplier->multiply(n, m, configuration.num_threads, matrix, vector, output);
            configuration.allocator->free(n, m, matrix, vector, output);
            configuration.times.push_back(time);
        }

        std::cout << configuration.toCsv() << std::endl;
    }
    return 0;
}