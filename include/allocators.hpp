#pragma once

#include <stdint.h>
#include <sys/mman.h>
#include <cassert>
#include <memory>
#include <string>

/// @brief Interface for an allocator that allocates a [n x m] matrix and a vector of size [m x 1].
class IMatrixVectorAllocator {
   public:
    typedef std::shared_ptr<IMatrixVectorAllocator> SharedPtr;

    virtual bool allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector,
                          float*& output) const = 0;

    virtual void free(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const = 0;

    virtual std::string getName() const = 0;
};

/// @brief Allocates the matrix in one contiguous block of memory, and the vector/outputs in their own blocks.
class DisjointMemoryAllocator : public IMatrixVectorAllocator {
    bool allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {

        if (n * m > 1 << 30) {
            return false;
        }

        vector = new float[m];
        output = new float[n];

        matrix = new float*[n];
        matrix[0] = new float[n * m];
        for (uint32_t i = 0; i < n; i++)
            matrix[i] = &matrix[0][i * m];

        return true;
    }

    void free(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        delete[] vector;
        delete[] output;
        delete[] matrix[0];
        delete[] matrix;
    }

    std::string getName() const override { return "DisjointMemoryAllocator"; }
};

/// @brief Allocates each row of the matrix in its own contiguous block of memory, and the vector/outputs in their own
class DisjointRowMemoryAllocator : public IMatrixVectorAllocator {
    bool allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        if (n * m > 1 << 30) {
            return false;
        }

        vector = new float[m];
        output = new float[n];

        matrix = new float*[n];
        for (uint32_t i = 0; i < n; i++)
            matrix[i] = new float[m];

        return true;
    }

    void free(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {

        delete[] vector;
        delete[] output;

        for (uint32_t i = 0; i < n; i++)
            delete[] matrix[i];
        delete[] matrix;
    }

    std::string getName() const override { return "DisjointRowMemoryAllocator"; }
};

/// @brief Memory allocator that allocates all of the different blocks in a _single_ contiguous memory block.
class ContiguousMemoryAllocator : public IMatrixVectorAllocator {
    bool allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        if (n * m > 1 << 30) {
            return false;
        }

        const uint64_t size = n * sizeof(float*) + (n * m + m + n) * sizeof(float);
        float* global = (float*)malloc(size);

        matrix = (float**)global;
        float* data_start = (float*)(global + n);
        matrix[0] = (float*)data_start;
        for (uint32_t i = 0; i < n; i++)
            matrix[i] = &matrix[0][i * m];

        vector = data_start + n * m;
        output = vector + m;

        return true;
    }

    void free(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        std::free(matrix);
    }

    std::string getName() const override { return "ContiguousMemoryAllocator"; }
};

/// @brief Memory allocator that uses mmap to allocate all of the different blocks in a _single_ contiguous memory block.
class MmapMemoryAllocator : public IMatrixVectorAllocator {
    bool allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        const uint64_t size = n * sizeof(float*) + (n * m + m + n) * sizeof(float);
        void* vmem_region = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (vmem_region == 0 || vmem_region == MAP_FAILED) {
            return false;
        }

        float* global = (float*)vmem_region;

        matrix = (float**)global;
        float* data_start = (float*)(global + n);
        matrix[0] = (float*)data_start;
        for (uint32_t i = 0; i < n; i++)
            matrix[i] = &matrix[0][i * m];

        vector = data_start + n * m;
        output = vector + m;
        return true;
    }

    void free(const uint32_t n, const uint32_t m, float**& matrix, float*& vector, float*& output) const override {
        const uint64_t size = n * sizeof(float*) + (n * m + m + n) * sizeof(float);
        assert(munmap(matrix, size) == 0 && "Failed to unmap memory region");
    }

    std::string getName() const override { return "MmapMemoryAllocator"; }
};