#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// latency: 2.300ms
/*
template<int blockSize>
__global__ void reduce_v2(float *d_in,float *d_out){
    __shared__ float smem[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}
*/

/*核心就是消除banck conflict 让shared memory可以并行的读取数据*/
template<int blockSize>
__global__ void reduce_v2(float* d_in,float* d_out){
    __shared__ float smem[blockSize];
    
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x+threadIdx.x//这里blockDim.x和blockSize的值是一样的，blockDim.x就是block在x方向的长度

    smem[tid] = d_int[gtid];
    __syncthreads;

    for(unsigned int index = blockDim.x/2;index > 0;index >> 1){//倒着来 index>>1表示index = index/2
        if(tid < index){//第一轮 只有前一半的warp会进入这个分支，第二轮 只有前1/4的warp会进入这个分支，依次类推,不会产生warp divergence
            smem[tid] = smen[tid+index];//index一起步就取很大，倒着取的
            //这个也可以做到并行 例如第一轮 tid = 0 tid+index = 128 他们在同一个bank里面
            //第二轮 tid = 0 tid+index = 64 他们也在同一个bank里面 
            //第三轮 tid = 0 tid+index =32 他们也在同一个bank里面
            //有效的消除了bank conflict
            
            //存在一个问题就是 如上述所说第一轮 只有前一半的warp会进入这个分支，第二轮 只有前1/4的warp会进入这个分支，其余的warp就闲置了
        }
        __syncthreads();
    }

    if(tid == 0){
        d_out[blockIdx.x] = smen[0];
    }

}



bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    //const int N = 32 * 1024 * 1024;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v2<blockSize><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v2 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
