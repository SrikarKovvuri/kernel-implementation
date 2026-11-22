// Direct tests of the actual C2 kernel: custom data and full default tensor.
#define main c2_driver_main
#include "../../src/partc/c2_tiled.cu"
#undef main
#include <vector>
#include <algorithm>

void check(int N,int d,int kind) {
    std::vector<double> Q(N*d),K(N*d),V(N*d),expected(N*d),actual(N*d+2,-987654.0);
    for(int i=0;i<N;++i) for(int t=0;t<d;++t) {
        size_t ix=(size_t)i*d+t;
        if(kind==0) {Q[ix]=qval(i,t);K[ix]=kval(i,t);V[ix]=vval(i,t);}
        else {
            // kind 1: later tiles repeatedly increase the row maximum.
            // kind 2: very large logits; kind 3: uniform softmax.
            Q[ix]=kind==3?0.0:1.0+(t%3)*0.1;
            K[ix]=(kind==2?1000.0:0.01)*i;
            V[ix]=((i*7+t*3)%23-11)/7.0;
        }
    }
    for(int i=0;i<N;++i) {
        std::vector<double> scores(N);double maximum=-INFINITY;
        for(int j=0;j<N;++j) {
            double dot=0;for(int t=0;t<d;++t)dot+=Q[(size_t)i*d+t]*K[(size_t)j*d+t];
            scores[j]=dot/sqrt((double)d);maximum=std::max(maximum,scores[j]);
        }
        double denom=0;for(double &s:scores){s=exp(s-maximum);denom+=s;}
        for(int j=0;j<N;++j) for(int t=0;t<d;++t)expected[(size_t)i*d+t]+=scores[j]/denom*V[(size_t)j*d+t];
    }
    double *q,*k,*v,*o;size_t bytes=(size_t)N*d*sizeof(double);
    CUDA_CHECK(cudaMalloc(&q,bytes));CUDA_CHECK(cudaMalloc(&k,bytes));CUDA_CHECK(cudaMalloc(&v,bytes));CUDA_CHECK(cudaMalloc(&o,bytes+2*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(q,Q.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(k,K.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(v,V.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(o,actual.data(),bytes+2*sizeof(double),cudaMemcpyHostToDevice));
    attn_tiled<<<N,128,(d+256)*sizeof(double)>>>(N,d,1.0/sqrt((double)d),q,k,v,o+1);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(),o,bytes+2*sizeof(double),cudaMemcpyDeviceToHost));
    double err=0;
    for(size_t i=0;i<expected.size();++i) {if(!std::isfinite(actual[i+1]))exit(2);err=std::max(err,fabs(actual[i+1]-expected[i]));}
    bool ok=err<1e-8 && actual.front()==-987654.0 && actual.back()==-987654.0;
    printf("CHECK,N=%d,d=%d,kind=%d,max_abs_error=%.12g,guards=%s,%s\n",N,d,kind,err,ok?"PASS":"FAIL",ok?"PASS":"FAIL");
    if(!ok)exit(3);
    CUDA_CHECK(cudaFree(q));CUDA_CHECK(cudaFree(k));CUDA_CHECK(cudaFree(v));CUDA_CHECK(cudaFree(o));
}
int main(){check(257,19,1);check(257,19,2);check(129,129,3);check(3,1024,1);check(2048,64,0);}
