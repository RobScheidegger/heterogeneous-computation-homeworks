#include "../matrix_multiplier.hpp"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

class BlockingMultiplier : public IMatrixMultiplier {
   public:
    BlockingMultiplier(const uint32_t block_size_i, const uint32_t block_size_j, const uint32_t block_size_k)
        : block_size_i(block_size_i), block_size_j(block_size_j), block_size_k(block_size_k) {}

    uint64_t multiply(const uint32_t N, const uint32_t M, const uint32_t K, const uint8_t n_threads, float** C,
                      float** A, float** B) const override {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_threads)
        for (uint32_t ii = 0; ii < N; ii += block_size_i) {
            for (uint32_t i = ii; i < MIN(N, ii + block_size_i); i += 1) {
                for (uint32_t jj = 0; jj < M; jj += block_size_j) {
                    for (uint32_t j = 0; j < MIN(M, jj + block_size_j); j += 1) {
                        for (uint32_t k = 0; k < K; k += block_size_k) {
                            for (uint32_t kk = k; kk < MIN(K, k + block_size_k); kk += 1) {
                                C[i][j] = C[i][j] + A[i][k] * B[k][j];
                            }
                        }
                    }
                }
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