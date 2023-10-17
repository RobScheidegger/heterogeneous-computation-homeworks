#include <iostream>

#include "gtest/gtest.h"

#include <memory>

#include "math_utils.hpp"
#include "matrix.hpp"
#include "matrix_multiplier.hpp"
#include "multipliers/default_multiplier.hpp"

TEST(MatrixMultiplication, SimpleMultiplicationTest) {

    std::vector<IMatrixMultiplier::SharedPtr> multipliers{std::make_shared<DefaultMultiplierIJK>()};

    const uint32_t n = 2;
    const uint32_t k = 2;
    const uint32_t m = 2;

    for (auto& multiplier : multipliers) {
        Matrix C(n, m);
        Matrix A(n, k);
        Matrix B(k, m);

        A.data[0][0] = 1;
        A.data[0][1] = 2;
        A.data[1][0] = 3;
        A.data[1][1] = 4;

        B.data[0][0] = 5;
        B.data[0][1] = 6;
        B.data[1][0] = 7;
        B.data[1][1] = 8;

        multiplier->multiply(n, m, k, 4, C.data, A.data, B.data);

        // Verify that the result is correct
        float C_0[] = {19, 22};
        float C_1[] = {43, 50};
        EXPECT_EQ(C.data[0][0], C_0[0]);
        EXPECT_EQ(C.data[0][1], C_0[1]);
        EXPECT_EQ(C.data[1][0], C_1[0]);
        EXPECT_EQ(C.data[1][1], C_1[1]);
    }
}

TEST(MatrixMultiplication, ComplexMultiplicationTest) {
    std::vector<IMatrixMultiplier::SharedPtr> multipliers{std::make_shared<DefaultMultiplierIJK>()};

    const uint32_t n = 4;
    const uint32_t k = 4;
    const uint32_t m = 3;

    for (auto& multiplier : multipliers) {
        Matrix C(n, m);
        Matrix A(n, k);
        Matrix B(k, m);

        // Do the same entry-by-entry initialization but for the new n, k, m
        A.data[0][0] = 1;
        A.data[0][1] = 2;
        A.data[0][2] = 3;
        A.data[0][3] = 4;
        A.data[1][0] = 5;
        A.data[1][1] = 6;
        A.data[1][2] = 7;
        A.data[1][3] = 8;
        A.data[2][0] = 9;
        A.data[2][1] = 10;
        A.data[2][2] = 11;
        A.data[2][3] = 12;
        A.data[3][0] = 13;
        A.data[3][1] = 14;
        A.data[3][2] = 15;
        A.data[3][3] = 16;

        B.data[0][0] = 1;
        B.data[0][1] = 2;
        B.data[0][2] = 3;
        B.data[1][0] = 4;
        B.data[1][1] = 5;
        B.data[1][2] = 6;
        B.data[2][0] = 7;
        B.data[2][1] = 8;
        B.data[2][2] = 9;
        B.data[3][0] = 10;
        B.data[3][1] = 11;
        B.data[3][2] = 12;

        multiplier->multiply(n, m, k, 4, C.data, A.data, B.data);

        // Verify that the result is correct
        float C_0[] = {70, 80, 90};
        float C_1[] = {158, 184, 210};
        float C_2[] = {246, 288, 330};
        float C_3[] = {334, 392, 450};

        EXPECT_EQ(C.data[0][0], C_0[0]);
        EXPECT_EQ(C.data[0][1], C_0[1]);
        EXPECT_EQ(C.data[0][2], C_0[2]);
        EXPECT_EQ(C.data[1][0], C_1[0]);
        EXPECT_EQ(C.data[1][1], C_1[1]);
        EXPECT_EQ(C.data[1][2], C_1[2]);
        EXPECT_EQ(C.data[2][0], C_2[0]);
        EXPECT_EQ(C.data[2][1], C_2[1]);
        EXPECT_EQ(C.data[2][2], C_2[2]);
        EXPECT_EQ(C.data[3][0], C_3[0]);
        EXPECT_EQ(C.data[3][1], C_3[1]);
        EXPECT_EQ(C.data[3][2], C_3[2]);
    }
}