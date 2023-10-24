#include "../matrix.hpp"
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

    std::string getName() const override { return "DefaultMultiplierIJK"; }
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

    std::string getName() const override { return "DefaultMultiplierIJKCached"; }
};

class DefaultMultiplierIJKTranspose : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

        // Transpose B so we iterate k in the right direction
        Matrix b_T_matrix{M, K};
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < K; j++) {
                b_T_matrix.data[i][j] = B[j][i];
            }
        }

        float** b_T = b_T_matrix.data;

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                for (uint32_t k = 0; k < K; k++) {
                    C[i][j] = C[i][j] + A[i][k] * b_T[j][k];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override { return "DefaultMultiplierIJKTranspose"; }
};

class DefaultMultiplierIJKTransposeCached : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

        // Transpose B so we iterate k in the right direction
        Matrix b_T_matrix{M, K};
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < K; j++) {
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

    std::string getName() const override { return "DefaultMultiplierIJKTransposeCached"; }
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

    std::string getName() const override { return "DefaultMultiplierJIK"; }
};