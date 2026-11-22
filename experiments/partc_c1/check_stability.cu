// Exercise the student's actual kernels with inputs beyond the given driver.
#define main provided_c1_main
#include "../../src/partc/c1_naive.cu"
#undef main
#include <algorithm>
#include <vector>

int main() {
    const int N=33, d=19;
    for (double magnitude : {1.0, 1000.0}) {
        std::vector<double> Q(N*d), K(N*d), V(N*d), P(N*N), O(N*d);
        for (int i=0;i<N*d;++i) {
            Q[i]=magnitude*((i*7%23)-11)/7.0;
            K[i]=magnitude*((i*11%29)-14)/9.0;
            V[i]=((i*13%31)-15)/5.0;
        }
        double *q,*k,*v,*p,*o;
        CUDA_CHECK(cudaMalloc(&q,N*d*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&k,N*d*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&v,N*d*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p,N*N*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&o,(N*d+2)*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(q,Q.data(),N*d*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k,K.data(),N*d*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v,V.data(),N*d*sizeof(double),cudaMemcpyHostToDevice));
        std::vector<double> guarded(N*d+2,-123456.0);
        CUDA_CHECK(cudaMemcpy(o,guarded.data(),guarded.size()*sizeof(double),cudaMemcpyHostToDevice));
        scores_kernel<<<dim3((N+15)/16,(N+15)/16),dim3(16,16)>>>(N,d,1.0/sqrt((double)d),q,k,p);
        CUDA_CHECK(cudaGetLastError());
        softmax_rows_kernel<<<(N+255)/256,256>>>(N,p);
        CUDA_CHECK(cudaGetLastError());
        output_kernel<<<dim3((d+15)/16,(N+15)/16),dim3(16,16)>>>(N,d,p,v,o+1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(P.data(),p,N*N*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(guarded.data(),o,guarded.size()*sizeof(double),cudaMemcpyDeviceToHost));
        if(guarded.front()!=-123456.0 || guarded.back()!=-123456.0) return 1;
        double maxerr=0, rowsumerr=0;
        for(int i=0;i<N;++i) {
            std::vector<double> scores(N);
            for(int j=0;j<N;++j) {
                double dot=0;
                for(int t=0;t<d;++t) dot+=Q[i*d+t]*K[j*d+t];
                scores[j]=dot/sqrt((double)d);
            }
            double m=*std::max_element(scores.begin(),scores.end()), denom=0, rowsum=0;
            for(double &s:scores) {s=exp(s-m);denom+=s;}
            for(int j=0;j<N;++j) {
                if(!std::isfinite(P[i*N+j]) || P[i*N+j]<0) return 2;
                rowsum+=P[i*N+j];
            }
            rowsumerr=std::max(rowsumerr,fabs(rowsum-1));
            for(int t=0;t<d;++t) {
                double expected=0;
                for(int j=0;j<N;++j) expected+=scores[j]/denom*V[j*d+t];
                double actual=guarded[1+i*d+t];
                if(!std::isfinite(actual)) return 3;
                maxerr=std::max(maxerr,fabs(actual-expected));
            }
        }
        bool ok=maxerr<1e-8 && rowsumerr<1e-12;
        printf("STABILITY,magnitude=%.0f,N=%d,d=%d,max_abs_error=%.12g,max_rowsum_error=%.12g,guards=PASS,%s\n",magnitude,N,d,maxerr,rowsumerr,ok?"PASS":"FAIL");
        if(!ok) return 4;
        CUDA_CHECK(cudaFree(q));CUDA_CHECK(cudaFree(k));CUDA_CHECK(cudaFree(v));CUDA_CHECK(cudaFree(p));CUDA_CHECK(cudaFree(o));
    }
}
