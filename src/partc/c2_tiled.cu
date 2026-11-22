#include "common.h"

// Part C, C2 -- completed tiled, FlashAttention-style attention.
//
// The naive C1 kernel writes the whole N x N score matrix to global memory
// and reads it back. This kernel must NOT materialize it. Compute the same
// output as C1 by processing the keys and values in tiles and carrying the
// softmax statistics across tiles, so the full row of scores is never stored
// at once. You decide what to stage in shared memory. Derive the update that
// lets you fold each new tile of scores into a running result without having
// first seen the whole row; the lecture's FlashAttention material is the
// reference for the idea, but the recurrence is yours to work out.
//
// Choose the block-to-row mapping and the tile width yourself, and place a
// barrier wherever one thread's shared-memory writes must be visible to the
// others before they read. The output must match C1 element by element.
//
// Implementation below: one query row per 128-thread block, 128-key tiles,
// shared query/weights/reduction scratch, and register output accumulators.
// main() launches the completed kernel inside the supplied event timer.
__global__ void attn_tiled(int N, int d, double scale,
                            const double *Q, const double *K, const double *V, double *O)
{
    // One 128-thread block owns one query row. Only the query, one tile
    // of scores/weights, and reduction scratch live in shared memory.
    // Output numerators stay in registers (up to d=1024, as specified).
    const int B = 128;
    const int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ double scratch[];
    double *query = scratch;
    double *weights = query + d;
    double *reduce = weights + B;
    __shared__ double runningMax, denominator, rescale;
    double accum[8] = {};
    for (int t = tid; t < d; t += B) query[t] = Q[(size_t)row*d+t];
    if (tid == 0) { runningMax = -INFINITY; denominator = 0.0; }
    __syncthreads();

    for (int base = 0; base < N; base += B) {
        const int key = base + tid;
        double score = -INFINITY;
        if (key < N) {
            double dot = 0.0;
            for (int t = 0; t < d; ++t) dot += query[t]*K[(size_t)key*d+t];
            score = scale*dot;
        }
        reduce[tid] = score;
        __syncthreads();
        for (int stride = B/2; stride; stride /= 2) {
            if (tid < stride) reduce[tid] = fmax(reduce[tid], reduce[tid+stride]);
            __syncthreads();
        }
        if (tid == 0) {
            const double nextMax = fmax(runningMax, reduce[0]);
            rescale = exp(runningMax-nextMax);
            runningMax = nextMax;
        }
        __syncthreads();
        // Invalid keys get zero weight and are never used to read V.
        const double weight = key < N ? exp(score-runningMax) : 0.0;
        weights[tid] = weight;
        reduce[tid] = weight;
        __syncthreads();
        for (int stride = B/2; stride; stride /= 2) {
            if (tid < stride) reduce[tid] += reduce[tid+stride];
            __syncthreads();
        }
        if (tid == 0) denominator = rescale*denominator + reduce[0];
        const int valid = N-base < B ? N-base : B;
        for (int t = tid, slot = 0; t < d; t += B, ++slot) {
            double contribution = 0.0;
            // Adjacent threads read adjacent V columns at each key.
            for (int j = 0; j < valid; ++j)
                contribution += weights[j]*V[(size_t)(base+j)*d+t];
            accum[slot] = rescale*accum[slot] + contribution;
        }
        // Finish consuming this tile before shared weights are overwritten.
        __syncthreads();
    }
    for (int t = tid, slot = 0; t < d; t += B, ++slot)
        O[(size_t)row*d+t] = accum[slot]/denominator;
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    double scale = 1.0 / sqrt((double)a.d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)a.N * a.d * sizeof(double);

    double *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // defined output until you launch your kernel

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // Launch one block per query row with query and tile reduction storage.
    // The supplied timing scaffold is unchanged.
    const int threads = 128;
    const size_t shmemBytes = (a.d + 2*threads)*sizeof(double);
    attn_tiled<<<a.N, threads, shmemBytes>>>(a.N, a.d, scale, dQ, dK, dV, dO);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)a.N * a.d);

    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)a.N * a.d <= 4096) dumpOutput(hO, (size_t)a.N * a.d);
    printf("RESULT,program=c2_tiled,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           a.N, a.d, (double)ms, cs);

    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
