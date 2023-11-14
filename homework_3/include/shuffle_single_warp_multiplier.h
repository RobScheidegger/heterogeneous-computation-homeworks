#pragma once

#include <cuda.h>

#include "./matrix_vector_multiplier.hpp"

class ShuffleSingleWarpMultiplier : public IMatrixVectorMultiplier {
   public:
    uint32_t multiply(float* const A, float* const x, float* y, int num_rows, int num_cols) const override;

    std::string getName() const override;
};