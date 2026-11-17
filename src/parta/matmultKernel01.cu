#include "matmultKernel.h"

#define SUBTILE (FOOTPRINT_SIZE / BLOCK_SIZE)  // = 2

__device__ float GetElement(const Matrix m, int row, int col) {
    return m.elements[row * m.stride + col];
}
__device__ void SetElement(Matrix m, int row, int col, float value) {
    m.elements[row * m.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix m, int row, int col, int size) {
    Matrix sub;
    sub.width = size;
    sub.height = size;
    sub.stride = m.stride;
    sub.elements = &m.elements[m.stride * size * row + size * col];
    return sub;
}

// TODO (Part A, Q4 -- 10 pts): write the improved tiled matmul kernel.
//
// The thread block stays BLOCK_SIZE x BLOCK_SIZE = 16x16 = 256 threads, but
// FOOTPRINT_SIZE is now 32, so each block produces a 32x32 tile of C and
// each thread is responsible for a 2x2 block of four output elements rather
// than one. Decide how the 256 threads cooperatively load the larger
// FOOTPRINT_SIZE x FOOTPRINT_SIZE shared-memory tiles, how each thread
// accumulates its four outputs across the K dimension, and where each
// result is written back.
//
// Why this helps: a 32x32 tile stages larger As/Bs blocks than the 16x16
// kernel, so each element loaded from global memory into shared memory is
// consumed by more multiply-accumulates before it is evicted, and the whole
// matrix is read from global memory fewer times. Quantify that reuse change
// in your report rather than assuming a particular factor.
//
// The driver zeros C before launch, so the unimplemented stub produces
// defined (wrong) output and fails the correctness check until you write it.
// The GetElement/SetElement/GetSubMatrix helpers above are provided
// infrastructure you may use.
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    // TODO (A4): compute this block's FOOTPRINT_SIZE x FOOTPRINT_SIZE tile of
    // C with each thread producing its 2x2 subtile. Stage A and B tiles in
    // shared memory, accumulate across the K dimension, synchronize where
    // needed, and write the results back to C.
    int blockRow = blockIdx.y;
    int blockCol= blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    int startCol = threadIdx.x * 2;
    int numTiles = A.width / FOOTPRINT_SIZE;

    for (int m = 0; m < numTiles; m++) {
        Matrix Asub = GetSubMatrix(A, blockRow, m, FOOTPRINT_SIZE); //get the ptr to the submatrix of A in whatver tile
        Matrix Bsub = GetSubMatrix(B, m, blockCol, FOOTPRINT_SIZE); //get the ptr to the submatrix of B in whatever tile

        __shared__ float sa[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
        __shared__ float sb[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

        //each block must load a larger tile than q3 so now we need 32x32. That means each thread needs to load more elements from global memory.

        

        
        

        

    }
    
}
