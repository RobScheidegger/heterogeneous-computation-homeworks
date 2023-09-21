#pragma once

#include <stdint.h>
#include <memory>
#include <string>

/// @brief Interface for an allocator that allocates a [n x m] matrix and a vector of size [m x 1].
class IMatrixVectorAllocator {
   public:
    typedef std::shared_ptr<IMatrixVectorAllocator> SharedPtr;

    virtual void allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector,
                          float*& output) const = 0;

    virtual void free(float**& matrix, float*& vector, float*& output) const = 0;

    virtual std::string getName() const = 0;
};

class DisjointMemoryAllocator : public IMatrixVectorAllocator {
    void allocate(const uint32_t n, const uint32_t m, float**& matrix, float*& vector,
                  float*& output) const override {

        vector = new float[m];
        output = new float[n];

        // Fill with random values
    }

    void free(float**& matrix, float*& vector, float*& output) const override {}

    std::string getName() const override { return "DisjointMemoryAllocator"; }
};

class DisjointRowMemoryAllocator {};
