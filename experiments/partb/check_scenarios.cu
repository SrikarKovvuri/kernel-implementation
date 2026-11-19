// Check the actual runScenario implementations on small, non-aligned lengths.
#define main assignment_main
#ifdef TEST_UNIFIED
#include "../../src/partb/q3_unified.cu"
#else
#include "../../src/partb/q2_malloc.cu"
#endif
#undef main
#include <cmath>

#define CHECK(call) do { cudaError_t err=(call); if(err!=cudaSuccess) { \
    fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 1; } } while(0)
int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    printf("DEVICE,name=%s,managedMemory=%d,concurrentManagedAccess=%d,pageableMemoryAccess=%d\n",
        prop.name,prop.managedMemory,prop.concurrentManagedAccess,prop.pageableMemoryAccess);
    for(long n: {1L,255L,256L,257L,513L}) {
        size_t bytes=(n+2)*sizeof(float);
        std::vector<float> hx(n+2),hy(n+2),hz(n+2,-12345.0f);
        for(long i=0;i<n+2;i++) { hx[i]=(float)(i%17-8); hy[i]=(float)(i%11-5); }
        for(int s=1;s<=3;s++) {
            float *x,*y,*z;
#ifdef TEST_UNIFIED
            CHECK(cudaMallocManaged(&x,bytes)); CHECK(cudaMallocManaged(&y,bytes)); CHECK(cudaMallocManaged(&z,bytes));
            for(long i=0;i<n+2;i++) {x[i]=hx[i];y[i]=hy[i];z[i]=-12345.0f;}
#else
            CHECK(cudaMalloc(&x,bytes)); CHECK(cudaMalloc(&y,bytes)); CHECK(cudaMalloc(&z,bytes));
            std::fill(hz.begin(),hz.end(),-12345.0f);
            CHECK(cudaMemcpy(x,hx.data(),bytes,cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(y,hy.data(),bytes,cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(z,hz.data(),bytes,cudaMemcpyHostToDevice));
#endif
            runScenario(s,n,z+1,x+1,y+1);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
#ifdef TEST_UNIFIED
            for(long i=0;i<n+2;i++) hz[i]=z[i];
#else
            CHECK(cudaMemcpy(hz.data(),z,bytes,cudaMemcpyDeviceToHost));
#endif
            bool ok=hz[0]==-12345.0f && hz[n+1]==-12345.0f;
            for(long i=1;i<=n;i++) ok=ok && hz[i]==hx[i]+hy[i];
            CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(z));
            printf("CHECK,n=%ld,scenario=%d,every_element_and_guards=%s\n",n,s,ok?"PASS":"FAIL");
            if(!ok) return 2;
        }
    }
    return 0;
}
