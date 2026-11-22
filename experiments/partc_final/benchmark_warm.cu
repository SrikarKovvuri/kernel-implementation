// Same kernels and GEMM mapping; separate harness isolates repeated execution.
#define main c1_main
#define softmax_rows_kernel c1_softmax
#include "../../src/partc/c1_naive.cu"
#undef main
#undef softmax_rows_kernel
#define main c2_main
#include "../../src/partc/c2_tiled.cu"
#undef main
#define main c3_main
#define softmax_rows_kernel c3_softmax
#include "../../src/partc/c3_cublas.cu"
#undef main
#undef softmax_rows_kernel
#include <vector>
#include <algorithm>

void pipeline(int which,int N,int d,double *q,double *k,double *v,double *s,double *o,cublasHandle_t h) {
    const double scale=1.0/sqrt((double)d),one=1.0,zero=0.0;
    if(which==0) {
        scores_kernel<<<dim3((N+15)/16,(N+15)/16),dim3(16,16)>>>(N,d,scale,q,k,s);
        c1_softmax<<<(N+255)/256,256>>>(N,s);
        output_kernel<<<dim3((d+15)/16,(N+15)/16),dim3(16,16)>>>(N,d,s,v,o);
    } else if(which==1) attn_tiled<<<N,128,(d+256)*sizeof(double)>>>(N,d,scale,q,k,v,o);
    else {
        CUBLAS_CHECK(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,N,N,d,&scale,k,d,q,d,&zero,s,N));
        c3_softmax<<<(N+255)/256,256>>>(N,s);
        CUBLAS_CHECK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,d,N,N,&one,v,d,s,N,&zero,o,d));
    }
    CUDA_CHECK(cudaGetLastError());
}
int main() {
    const int N=2048,d=64;AttnDims dims{N,d};size_t bytes=(size_t)N*d*sizeof(double);
    double *Q=makeQ(dims),*K=makeK(dims),*V=makeV(dims),*q,*k,*v,*s,*o;
    CUDA_CHECK(cudaMalloc(&q,bytes));CUDA_CHECK(cudaMalloc(&k,bytes));CUDA_CHECK(cudaMalloc(&v,bytes));CUDA_CHECK(cudaMalloc(&s,(size_t)N*N*sizeof(double)));CUDA_CHECK(cudaMalloc(&o,bytes));
    CUDA_CHECK(cudaMemcpy(q,Q,bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(k,K,bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(v,V,bytes,cudaMemcpyHostToDevice));
    cublasHandle_t h;CUBLAS_CHECK(cublasCreate(&h));cudaEvent_t start,stop;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));
    const char *names[]={"c1_naive","c2_tiled","c3_cublas"};std::vector<double> actual(N*d),expected(N*d,0.0);
    // Independent CPU reference, computed once, outside all GPU timing.
    for(int i=0;i<N;++i) {
        std::vector<double> weights(N);double m=-INFINITY;
        for(int j=0;j<N;++j){double dot=0;for(int t=0;t<d;++t)dot+=Q[(size_t)i*d+t]*K[(size_t)j*d+t];weights[j]=dot/sqrt((double)d);m=std::max(m,weights[j]);}
        double denom=0;for(double &w:weights){w=exp(w-m);denom+=w;}
        for(int j=0;j<N;++j)for(int t=0;t<d;++t)expected[(size_t)i*d+t]+=weights[j]/denom*V[(size_t)j*d+t];
    }
    for(int round=-3;round<10;++round) for(int offset=0;offset<3;++offset) {
        const int which=(offset+(round<0?0:round))%3;
        CUDA_CHECK(cudaEventRecord(start));pipeline(which,N,d,q,k,v,s,o,h);CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        CUDA_CHECK(cudaMemcpy(actual.data(),o,bytes,cudaMemcpyDeviceToHost));
        double err=0;for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))return 2;err=std::max(err,fabs(actual[i]-expected[i]));}if(err>1e-8)return 3;
        printf("BENCH,program=%s,round=%d,phase=%s,N=%d,d=%d,time_ms=%.6f,checksum=%.9f,max_abs_error=%.12g\n",names[which],round,round<0?"warmup":"measured",N,d,(double)ms,checksum(actual.data(),actual.size()),err);
    }
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(stop));CUBLAS_CHECK(cublasDestroy(h));free(Q);free(K);free(V);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(s);cudaFree(o);
}
