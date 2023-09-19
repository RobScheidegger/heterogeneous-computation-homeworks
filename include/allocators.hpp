#pragma once

#include <stdint.h>
#include <memory>

/// @brief Interface for an allocator that allocates a [n x m] matrix and a vector of size [m x 1].
class IMatrixVectorAllocator {
   public:
    typedef std::shared_ptr<IMatrixVectorAllocator> SharedPtr;

    virtual void allocate(const uint32_t n, const uint32_t m, float**& matrix,
                          float*& vector, float*& output) = 0;

    virtual void free(float**& matrix, float*& vector, float*& output) = 0;
};

class DisjointMemoryAllocator {};

class DisjointRowMemoryAllocator {};
