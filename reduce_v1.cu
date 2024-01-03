#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//#define THREAD_PER_BLOCK 256
//屏蔽掉blockSize带来的forloop的编译优化为3.065ms
// bank conflict
/*
template<int blockSize>
__global__ void reduce_v1(float *d_in,float *d_out){
    __shared__ float smem[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}
*/


/*核心思想就是让warp的每一个thread都可以执行相同的分支，有效的消除warp divergence*/
/*实现的方案 我个人总结就是 用连续的tid去映射离散的smem -> int index = 2 * s * tid;*/
template<int blockSize>
__global__ void reduce_v1(float* d_in,float* d_out){
    __shared__ float smem[blockSize];//每个block分配一个sharedmemory，sharedmemory的float个数就等于blocSize

    unsigned int tid = threadIdx.x;//定义一个blcok内的id
    unsigned int gtid = blockDim.x*blockIdx.x + threadIdx.x;//定义一个全局id

    smem[tid] =  d_int[gtid];
    __syncthreads();//等待线程同步，这里是所有线程都进行的一个读取操作！

    for(unsigned int s = 1;s < blockDimx.x; s *= 2){
        int index = 2 * s * tid;
        if(index < blockDim.x){
            smen[index] += smem[index+s];//index正着取的
        }
        __syncthreads();
        //第一轮 index = 2*tid < 256(blockSize = 256) -> tid<128 只计算tid<128的索引 三个warp
        //第一轮 0-1 2-3 4-5 6-7(相当于 0号线程操作0号sharedmemoy 1号线程操作2号sharedmemory)

        //第二轮 index = 4*tid < 256 ->tid<64 只计算tid<64的索引 两个warp
        //第二轮 0-2 4-6(相当于0号线程操作0号sharedmemory  1号线程操作4号sharedmemory)

        //第三轮 只计算tid<32索引 一个warp

        //前三轮里面调度了的warp都是满线程在工作，避免了warp divergence
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
    reduce_v1<blockSize><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
