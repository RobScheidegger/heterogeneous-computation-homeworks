#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "math_utils.hpp"
#include "matrix.hpp"

#define WARMUP_OPERATIONS 100000
#define BENCHMARK_REPITITIONS 4

struct BenchmarkConfiguration {
    uint32_t n;
    uint32_t m;
    uint32_t repetitions = BENCHMARK_REPITITIONS;

    std::vector<uint64_t> times;

    BenchmarkConfiguration(uint32_t n, uint32_t m) : n(n), m(m) {}

    std::string toCsv() const {
        auto meanStdDev = getMeanAndStdDev<uint64_t, float>(times);

        return std::to_string(n) + ',' + std::to_string(m) + ',' + std::to_string(repetitions) + ',' +
               std::to_string(meanStdDev.first) + ',' + std::to_string(meanStdDev.second);
    }
};

int main(int argc, char** argv) {
    std::vector<uint32_t> m_options{10, 100, 1000, 10000};
    std::vector<float> aspect_ratio_options{0.01, 0.1, 1, 10, 100};
    std::vector<BenchmarkConfiguration> configurations;

    std::cout << "Creating benchmark configurations..." << std::endl;

    for (auto& m : m_options) {
        for (auto& aspect_ratio : aspect_ratio_options) {
            configurations.emplace_back(BenchmarkConfiguration((uint32_t)m * aspect_ratio, m));
        }
    }

    std::cout << "Found " << configurations.size() << " configurations." << std::endl;

    uint32_t experiment_number = 1;
    for (auto& configuration : configurations) {
        const uint32_t n = configuration.n;
        const uint32_t m = configuration.m;

        uint32_t garbage = 1;
        for (uint32_t i = 1; i < WARMUP_OPERATIONS; i++) {
            garbage *= i;
        };

        for (uint32_t repetition = 0; repetition < configuration.repetitions; repetition++) {
            Matrix A(n, m);
            std::vector<float> x;
            std::vector<float> y;

            A.randomize();
            for (uint32_t i = 0; i < m; i++) {
                x.push_back((float)(rand() % 65536) - 32768);
            }
            for (uint32_t i = 0; i < n; i++) {
                y.push_back((float)(rand() % 65536) - 32768);
            }

            // perform the memcopy to the device

            // start timer

            // call the kernel

            // end timer
            const uint64_t time = 0;

            configuration.times.push_back(time);
        }

        std::cout << experiment_number << "," << configuration.toCsv() << std::endl;
        experiment_number++;
    }
}