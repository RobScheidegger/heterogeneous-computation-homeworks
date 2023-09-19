#include <stdio.h>
#include <memory>

#include "allocators.hpp"

#define WARMUP_PERIOD_MS 100

struct BenchmarkConfiguration {
    uint32_t n;
    uint32_t m;
    uint32_t num_threads;
    IMatrixVectorAllocator::SharedPtr allocator;

    BenchmarkConfiguration(const uint32_t n, const uint32_t m,
                           const uint32_t num_threads,
                           IMatrixVectorAllocator::SharedPtr allocator)
        : n(n), m(m), num_threads(num_threads), allocator(allocator) {}
};

struct BenchmarkResult {};

int main(int argc, char** argv) {
    printf("Hello world!\n");
    return 0;
}