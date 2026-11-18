// FP32 comparison of the unchanged assignment kernels and cuBLAS SGEMM.
// Original kernel symbols are renamed at compile time to link both variants.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "matmultKernel.h"

__global__ void MatMul00(const Matrix, const Matrix, Matrix);
__global__ void MatMul01(const Matrix, const Matrix, Matrix);

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
    fprintf(stderr,"CUDA error: %s, line %d\n",cudaGetErrorString(e),__LINE__); exit(1); } } while(0)
#define BLAS_OK(x) do { cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS error: %d, line %d\n",(int)e,__LINE__); exit(1); } } while(0)

int main() {
    const int n=1024, warmups=10, trials=50;
    const size_t count=(size_t)n*n, bytes=count*sizeof(float);
    std::vector<float> a(count),b(count),ref(count),output(count);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        a[(size_t)i*n+j]=(float)((i+j)%13);
        b[(size_t)i*n+j]=(float)((i-j+n)%7);
    }
    // Same independent CPU reference calculation as the assignment driver.
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        float sum=0;
        for(int k=0;k<n;k++) sum+=a[(size_t)i*n+k]*b[(size_t)k*n+j];
        ref[(size_t)i*n+j]=sum;
    }
    Matrix A={n,n,n,nullptr},B={n,n,n,nullptr};
    Matrix C[3]={{n,n,n,nullptr},{n,n,n,nullptr},{n,n,n,nullptr}};
    CUDA_OK(cudaMalloc(&A.elements,bytes));
    CUDA_OK(cudaMalloc(&B.elements,bytes));
    CUDA_OK(cudaMemcpy(A.elements,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B.elements,b.data(),bytes,cudaMemcpyHostToDevice));
    for(auto &c:C) { CUDA_OK(cudaMalloc(&c.elements,bytes)); CUDA_OK(cudaMemset(c.elements,0,bytes)); }
    cublasHandle_t handle;
    BLAS_OK(cublasCreate(&handle));
    // Prevent TF32/reduced-precision arithmetic for a like-for-like FP32 test.
    BLAS_OK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    int version=0; BLAS_OK(cublasGetVersion(handle,&version));
    printf("CONFIG,n=%d,warmups=%d,trials=%d,cublas_version=%d,math=FP32_PEDANTIC\n",n,warmups,trials,version);
    const char* names[]={"matmult00","matmult01","cublas_sgemm"};
    const float alpha=1.0f,beta=0.0f;
    auto run=[&](int impl) {
        if(impl==0) MatMul00<<<dim3(n/16,n/16),dim3(16,16)>>>(A,B,C[0]);
        else if(impl==1) MatMul01<<<dim3(n/32,n/32),dim3(16,16)>>>(A,B,C[1]);
        else {
            // Row-major C=A*B is column-major C^T=B^T*A^T: reverse operands.
            BLAS_OK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
                &alpha,B.elements,n,A.elements,n,&beta,C[2].elements,n));
        }
        CUDA_OK(cudaGetLastError());
    };
    auto validate=[&](const char* phase) {
        CUDA_OK(cudaDeviceSynchronize());
        for(int impl=0;impl<3;impl++) {
            CUDA_OK(cudaMemcpy(output.data(),C[impl].elements,bytes,cudaMemcpyDeviceToHost));
            double max_error=0;
            bool correct=true;
            for(size_t i=0;i<count;i++) {
                double error=fabs((double)output[i]-ref[i]);
                if(!std::isfinite(output[i]) || error>1e-2) correct=false;
                max_error=std::max(max_error,error);
            }
            printf("CHECK,phase=%s,program=%s,correct=%d,max_abs_err=%.9g\n",phase,names[impl],correct?1:0,max_error);
            if(!correct) exit(2);
        }
    };
    // Warm all implementations inside the same process before any timings.
    for(int w=0;w<warmups;w++) for(int impl=0;impl<3;impl++) run(impl);
    validate("after_warmup");
    cudaEvent_t start,stop;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    std::mt19937 random(6998);
    for(int trial=1;trial<=trials;trial++) {
        std::vector<int> order={0,1,2}; std::shuffle(order.begin(),order.end(),random);
        for(int impl:order) {
            CUDA_OK(cudaEventRecord(start)); run(impl); CUDA_OK(cudaEventRecord(stop));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,start,stop));
            printf("RESULT,program=%s,trial=%d,time_ms=%.6f\n",names[impl],trial,(double)ms);
        }
    }
    validate("after_measurement");
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    BLAS_OK(cublasDestroy(handle));
    for(auto &c:C) CUDA_OK(cudaFree(c.elements));
    CUDA_OK(cudaFree(A.elements)); CUDA_OK(cudaFree(B.elements));
    return 0;
}
