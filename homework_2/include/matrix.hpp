#include <cstdint>
#include <cstring>

class Matrix {
   public:
    Matrix(const uint32_t n, const uint32_t m) : n(n), m(m) {
        data = new float*[n];
        data[0] = new float[n * m];
        for (uint32_t i = 0; i < n; i++)
            data[i] = &data[0][i * m];

        // Make sure we start with zeroed entries
        std::memset(data[0], 0, n * m * sizeof(float));
    }

    ~Matrix() {
        delete[] data[0];
        delete[] data;
    }

    /// @brief Randomizes the data in the matrix with arbitrary floats between -65536 and 65536
    void randomize() {
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < m; j++)
                data[i][j] = (float)(rand() % 65536) - 32768;
    }

    float** data;
    const uint32_t n;
    const uint32_t m;
};