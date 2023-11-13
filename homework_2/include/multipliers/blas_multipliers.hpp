#include "../matrix_multiplier.hpp"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

class CBlasMultiplier : public IMatrixMultiplier {
   public:
    BlockingMultiplier(const uint32_t block_size_i, const uint32_t block_size_j, const uint32_t block_size_k)
        : block_size_i(block_size_i), block_size_j(block_size_j), block_size_k(block_size_k) {}

    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads) collapse(2)
        for (uint32_t i = 0; i < N; i += 1) {
            for (uint32_t j = 0; j < M; j += 1) {
                float cij = C[i][j];
                for (uint32_t k = 0; k < K; k += block_size_k) {
                    for (uint32_t kk = k; kk < MIN(K, k + block_size_k); kk += 1) {
                        cij += A[i][k] * B[k][j];
                    }
                }
                C[i][j] = cij;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::string getName() const override {
        return "BlockingMultiplier(" + std::to_string(block_size_i) + "." + std::to_string(block_size_j) + "." +
               std::to_string(block_size_k) + ")";
    }

   private:
    const uint32_t block_size_i;
    const uint32_t block_size_j;
    const uint32_t block_size_k;
};