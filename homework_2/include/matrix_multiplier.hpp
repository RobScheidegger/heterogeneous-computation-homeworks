#pragma once

#include <stdint.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

/// @brief Interface for a matrix multiplier that multiplies two matrices A and B and stores the result in C.
class IMatrixMultiplier {
   public:
    typedef std::shared_ptr<IMatrixMultiplier> SharedPtr;

    virtual uint64_t multiply(const uint32_t n, const uint32_t m, const uint32_t k, const uint8_t n_threads, float** C,
                              float** A, float** B) const = 0;

    virtual std::string getName() const = 0;
};
