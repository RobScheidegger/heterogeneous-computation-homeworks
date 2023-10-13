#pragma once

#include <stdint.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

/// @brief Interface for an allocator that allocates a [n x m] matrix and a vector of size [m x 1].
class IMultiplier {
   public:
    typedef std::shared_ptr<IMultiplier> SharedPtr;

    virtual uint64_t multiply(const uint32_t n, const uint32_t m, const uint8_t n_threads, float** matrix,
                              float* vector, float* output) const = 0;

    virtual std::string getName() const = 0;
};

class RowColumnMultiplier : public IMultiplier {
   public:
    uint64_t multiply(const uint32_t n, const uint32_t m, const uint8_t n_threads, float** matrix, float* vector,
                      float* output) const override {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < m; j++) {
                output[i] += matrix[i][j] * vector[j];
            }
        }
        // Return the time difference
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    std::string getName() const override {
        return "RowColumnMultiplier";
    }
};

class ColumnRowMultiplier : public IMultiplier {
   public:
    uint64_t multiply(const uint32_t n, const uint32_t m, const uint8_t n_threads, float** matrix, float* vector,
                      float* output) const override {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(n_threads)
        for (uint32_t j = 0; j < m; j++) {
            for (uint32_t i = 0; i < n; i++) {
                output[i] += matrix[i][j] * vector[j];
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    std::string getName() const override {
        return "ColumnRowMultiplier";
    }
};
