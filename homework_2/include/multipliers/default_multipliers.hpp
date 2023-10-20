#include "../matrix_multiplier.hpp"

class DefaultMultiplierIJK : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                for (uint32_t k = 0; k < K; k++) {
                    C[i][j] = C[i][j] + A[i][k] * B[k][j];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "DefaultMultiplierIJK";
    }
};

class DefaultMultiplierIJKCached : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                float cij = C[i][j];
                for (uint32_t k = 0; k < K; k++) {
                    cij = cij + A[i][k] * B[k][j];
                }
                C[i][j] = cij;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "DefaultMultiplierIJKCached";
    }
};

class DefaultMultiplierIJKTranspose : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

        // Transpose B so we iterate k in the right direction
        Matrix b_T_matrix{K, M};
        for (uint32_t i = 0; i < K; i++) {
            for (uint32_t j = 0; j < M; j++) {
                b_T_matrix.data[i][j] = B[j][i];
            }
        }

        float** b_T = b_T_matrix.data;

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                float cij = C[i][j];
                for (uint32_t k = 0; k < K; k++) {
                    cij = cij + A[i][k] * b_T[j][k];
                }
                C[i][j] = cij;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "DefaultMultiplierIJKTranspose";
    }
};

class DefaultMultiplierIJKTranspose1D : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

        // Transpose B so we iterate k in the right direction
        Matrix b_T_matrix{K, M};
        for (uint32_t i = 0; i < K; i++) {
            for (uint32_t j = 0; j < M; j++) {
                b_T_matrix.data[i][j] = B[j][i];
            }
        }

        float** b_T = b_T_matrix.data;

        float* C_1 = C[0];
        const float* A_1 = A[0];
        const float* B_1 = b_T[0];

        float cij;
#pragma omp parallel for num_threads(n_threads) collapse(2)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                cij = C_1[i * N + j];
                const float* a_i = &A_1[i * M];
                const float* b_j = &B_1[j * N];
                for (uint32_t k = 0; k < K; k++) {
                    cij = cij + (*(a_i + k)) * (*(b_j + k));
                }
                C_1[i * N + j] = cij;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "DefaultMultiplierIJKTranspose1D";
    }
};

class DefaultMultiplierJIK : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t j = 0; j < M; j++) {
            for (uint32_t i = 0; i < N; i++) {
                for (uint32_t k = 0; k < K; k++) {
                    C[i][j] = C[i][j] + A[i][k] * B[k][j];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "DefaultMultiplierJIK";
    }
};