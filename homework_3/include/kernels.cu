#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define ROWS_PER_BLOCK 5

__global__ void saxby_mult_warps(float* A, float* x, int num_rows, int num_cols, float* y) {
    int matrix_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute the dot product between A[matrix_row_idx : matrix_row_idx + num_cols] and x
    // (recall that A is a 1D flattened vector representing a matrix)

    // telling us which part of the column in A we're in for the given matrix_row_id
    int id = threadIdx.y + blockIdx.y * blockDim.y;
    // int block_id = blockIdx.y;

    int nwarps = blockDim.y / WARP_SIZE;
    int warp_id = threadIdx.y / WARP_SIZE;
    int lane_id = threadIdx.y % WARP_SIZE;

    // Shared memory is local to the thread block, so we don't need to worry about other blocks overwritting this
    __shared__ float shared_dot_product[BLOCK_SIZE / WARP_SIZE];

    float sum = 0.0f;
    if (id < num_cols) {
        sum = A[matrix_row_idx * num_cols + id] * x[id];
    }
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffff, sum, offset);

    if (lane_id == 0)
        shared_dot_product[warp_id] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int i = 0; i < nwarps; i++) {
            y[matrix_row_idx] += shared_dot_product[i];
        }
    }
}

__global__ void saxby_mult_blocks_per_row(float* A, float* x, int num_rows, int num_cols, float* y) {
    int matrix_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute the dot product between A[matrix_row_idx : matrix_row_idx + num_cols] and x
    // (recall that A is a 1D flattened vector representing a matrix)

    // telling us which part of the column in A we're in for the given matrix_row_id
    int id = threadIdx.y + blockIdx.y * blockDim.y;
    // int block_id = blockIdx.y;

    int nwarps = blockDim.y / WARP_SIZE;
    int warp_id = threadIdx.y / WARP_SIZE;
    int lane_id = threadIdx.y % WARP_SIZE;

    // Shared memory is local to the thread block, so we don't need to worry about other blocks overwritting this
    __shared__ float shared_dot_product[BLOCK_SIZE / (WARP_SIZE * ROWS_PER_BLOCK)];

    float sum = 0.0f;
    if (id < num_cols) {
        sum = A[matrix_row_idx * num_cols + id] * x[id];
    }
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffff, sum, offset);

    if (lane_id == 0)
        shared_dot_product[warp_id] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int i = 0; i < nwarps; i++) {
            y[matrix_row_idx] += shared_dot_product[i];
        }
    }
}

// Kernels required
// Single Warp Per Row
// Multiple Warps Per Row
// Multiple Rows per Thread Block

// Improved Version For Wide Columns