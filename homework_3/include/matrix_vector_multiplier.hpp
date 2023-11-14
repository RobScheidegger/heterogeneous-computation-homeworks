#pragma once

#include <stdint.h>
#include <chrono>
#include <memory>
#include <string>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a < b ? b : a)
#define CEIL(x, y) ((x + y - 1) / y)

#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32

class IMatrixVectorMultiplier {
   public:
    typedef std::shared_ptr<IMatrixVectorMultiplier> SharedPtr;

    virtual uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const = 0;

    virtual std::string getName() const = 0;
};
