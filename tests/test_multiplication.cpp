#include <iostream>

#include "gtest/gtest.h"

#include "allocators.hpp"
#include "math_utils.hpp"
#include "multipliers.hpp"

TEST(Example, Test) {

    RowColumnMultiplier rowMultiplier = RowColumnMultiplier();

    const uint32_t n = 3;
    const uint32_t m = 3;

    float m1[] = {7, 4, 2};
    float m2[] = {6, 5, 1};
    float m3[] = {10, 9, 3};

    float* matrix[3] = {m1, m2, m3};
    float vector[] = {1, 2, 3};
    float actualOutput[] = {0, 0, 0};
    float expectedOutput[] = {21, 19, 37};

    rowMultiplier.multiply(n, m, 1, matrix, vector, actualOutput);

    for (uint32_t i = 0; i < n; i++) {
        std::cout << "actual: " << actualOutput[i] << " expected: " << expectedOutput[i] << std::endl;
        EXPECT_EQ(actualOutput[i], expectedOutput[i]);
    }
}