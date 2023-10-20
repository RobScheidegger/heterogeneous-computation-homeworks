
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
#include "multipliers/blocking_multipliers.hpp"
#include "multipliers/default_multipliers.hpp"
#include "multipliers/openmp_multipliers.hpp"

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPETITIONS 4

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
        : n(n), m(m), k(k), num_threads(num_threads), multiplier(multiplier) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(k) + ',' +
               std::to_string(num_threads) + ',' + multiplier->getName() + ',' + std::to_string(repetitions) + ',' +
               std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    };
};

int main(int argc, char** argv) {
    std::vector<uint32_t> n_options{10, 100, 1000, 5000};
    std::vector<uint32_t> m_options{10, 100, 1000, 5000};
    std::vector<uint32_t> k_options{10, 100, 1000, 5000};
    std::vector<uint32_t> num_threads_options{4, 16};
    std::vector<uint32_t> blocking_options{
        16, 32, 64, 128, 256,
    };

    // Check CPU affinity
    int* cpuid = new int[omp_get_max_threads()];
#pragma omp parallel
    { cpuid[omp_get_thread_num()] = sched_getcpu(); }

    for (int k = 0; k < omp_get_max_threads(); k++)
        std::cout << k << "," << cpuid[k] << std::endl;

    std::vector<IMatrixMultiplier::SharedPtr> multipliers;
    if (argc == 2 && strcmp(argv[1], "--blocking") == 0) {
        for (const uint32_t block_size_i : blocking_options) {
            for (const uint32_t block_size_j : blocking_options) {
                for (const uint32_t block_size_k : blocking_options) {
                    multipliers.push_back(
                        std::make_shared<BlockingMultiplier>(block_size_i, block_size_j, block_size_k));
                }
            }
        }
        n_options.clear();
        m_options.clear();
        k_options.clear();

        n_options.push_back(100);
        m_options.push_back(100);
        k_options.push_back(1000);
    } else if (argc == 2 && strcmp(argv[1], "--baseline") == 0) {
        multipliers = {std::make_shared<DefaultMultiplierIJKCached>(),
                       std::make_shared<DefaultMultiplierIJKTranspose>(),
                       std::make_shared<DefaultMultiplierIJKTranspose1D>()};
        n_options = {100};
        m_options = {100};
        k_options = {1000};

    } else {
        multipliers = {std::make_shared<DefaultMultiplierIJK>(), std::make_shared<DefaultMultiplierIJKCached>(),
                       std::make_shared<DefaultMultiplierJIK>(), std::make_shared<Collapse2Multiplier>(),
                       std::make_shared<Collapse3Multiplier>()};
    }

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
    uint32_t experiment_number = 0;
    for (auto& configuration : configurations) {
        experiment_number++;

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

        std::cout << experiment_number << "," << configuration.toCsv() << std::endl;
    }
    return 0;
}