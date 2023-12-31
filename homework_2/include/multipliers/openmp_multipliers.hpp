#include "../matrix_multiplier.hpp"

class Collapse2Multiplier : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads) collapse(2)
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
        return "Collapse2Multiplier";
    }
};

class Collapse3Multiplier : public IMatrixMultiplier {
   public:
    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

        // using collapse(3) means that we can't actualy cache the C_{ij} value and update after.
#pragma omp parallel for num_threads(n_threads) collapse(3)
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                for (uint32_t k = 0; k < K; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "Collapse3Multiplier";
    }
};