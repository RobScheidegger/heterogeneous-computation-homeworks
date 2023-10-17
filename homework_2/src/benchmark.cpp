
#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "math_utils.hpp"
#include "matrix.hpp"
#include "matrix_multiplier.hpp"

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPETITIONS 10

struct BenchmarkConfiguration {
    // Configuration fields
    uint32_t n;
    uint32_t m;
    uint32_t k;
    uint32_t num_threads;
    IMatrixMultiplier::SharedPtr multiplier;
    uint32_t repetitions = BENCHMARK_REPETITIONS;

    // Result fields
    std::vector<uint64_t> times;

    BenchmarkConfiguration(const uint32_t n, const uint32_t m, const uint32_t k, const uint32_t num_threads,
                           IMatrixMultiplier::SharedPtr multiplier)
        : n(n), m(m), num_threads(num_threads), multiplier(multiplier) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(k) + ',' +
               std::to_string(num_threads) + ',' + multiplier->getName() + ',' + std::to_string(repetitions) + ',' +
               std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    };
};

const std::vector<uint32_t> n_options{1, 10, 100, 1000, 10000, 25000};
const std::vector<uint32_t> m_options{1, 10, 100, 1000, 10000, 25000};
const std::vector<uint32_t> k_options{1, 10, 100, 1000, 10000, 25000};
const std::vector<uint32_t> num_threads_options{1, 4, 16};

int main(int argc, char** argv) {
    // Check CPU affinity
    int* cpuid = new int[omp_get_max_threads()];
#pragma omp parallel
    { cpuid[omp_get_thread_num()] = sched_getcpu(); }

    for (int k = 0; k < omp_get_max_threads(); k++)
        std::cout << k << "," << cpuid[k] << std::endl;

    std::vector<IMatrixMultiplier::SharedPtr> multipliers{};

    std::vector<BenchmarkConfiguration> configurations;

    std::cout << "Creating benchmark configurations..." << std::endl;

    // Make all of the required configurations
    for (auto& n : n_options) {
        for (auto& m : m_options) {
            for (auto& k : k_options) {
                for (auto& num_threads : num_threads_options) {
                    for (auto& multiplier : multipliers) {
                        configurations.emplace_back(n, m, k, num_threads, multiplier);
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
        const uint32_t k = configuration.k;

        // Warmup with some garbage computation
        uint32_t garbage = 1;
        for (uint32_t i = 1; i < WARMUP_OPERATIONS; i++) {
            garbage *= i;
        };

        for (uint32_t iteration = 0; iteration < configuration.repetitions; iteration++) {

            Matrix C{n, m};
            Matrix A{n, k};
            Matrix B{k, m};

            // Randomly initialize A and B
            C.randomize();
            A.randomize();
            B.randomize();

            const uint64_t time =
                configuration.multiplier->multiply(n, m, k, configuration.num_threads, C.data, A.data, B.data);
            configuration.times.push_back(time);
        }

        std::cout << configuration.toCsv() << std::endl;
    }
    return 0;
}