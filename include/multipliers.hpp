#pragma once

#include <stdint.h>
#include <chrono>
#include <memory>
#include <string>

/// @brief Interface for an allocator that allocates a [n x m] matrix and a vector of size [m x 1].
class IMultiplier {
   public:
    typedef std::shared_ptr<IMultiplier> SharedPtr;

    virtual uint64_t multiply(const uint32_t n, const uint32_t m, float** matrix, float* vector,
                              float* output) const = 0;

    virtual std::string getName() const = 0;
};

class RowColumnMultiplier : public IMultiplier {
    uint64_t multiply(const uint32_t n, const uint32_t m, float** matrix, float* vector, float* output) const override {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // Get the time
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < m; j++) {
                output[i] += matrix[i][j] * vector[j];
            }
        }
        // Return the time difference
        std::chrono::steady_clock::time_point endLine = std::chrono::steady_clock::now();

        return std::chrono::duration_cast
    }

    std::string getName() const override { return "RowColumnMultiplier"; }
};

class ColumnRowMultiplier : public IMultiplier {
    uint64_t multiply(const uint32_t n, const uint32_t m, float** matrix, float* vector, float* output) const override {
    }

    std::string getName() const override { return "ColumnRowMultiplier"; }
};
