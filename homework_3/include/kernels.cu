__global__ void matrixVectorAdd(float* y, const float* A, const float* x, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        float sum = 0.0f;
        for (int col = 0; col < m; ++col) {
            sum += A[row * m + col] * x[col];
        }
        y[row] += sum;
    }
}

// Kernels required
// Single Warp Per Row
// Multiple Warps Per Row
// Multiple Rows per Thread Block

// Improved Version For Wide Columns