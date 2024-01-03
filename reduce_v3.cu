#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//latency: 1.147ms
/*
template<int blockSize>
__global__ void reduce_v3(float *d_in, float *d_out){
    __shared__ float smem[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
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
}*/


/*优化的核心点是启动一半的线程，让一个线程处理两个block(奇block和偶block)对应位置的元素！！
以前第一轮的空闲线程是blockSize/2，现在优化了之后是blockSize/4
为什么呢？
因为现在只启动了blockSize/2个线程，第一轮只有blockSize/2的一半会参与运算，相当于第一轮只有blockSize/4会参与运算
*/
template<int blockSize>
__global__ void reduce_v3(floast* d_int,float* d_out){
    __shared__ float smen[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;//只取0 2 4号block里面的global thread id，相当于第一个 第三个 第五个
    //计算 gtid 的方式是将当前线程块的起始全局索引（blockIdx.x * (blockSize * 2)）与当前线程在线程块内的索引相加。
    //这样，gtid 就表示了当前线程负责处理的两个相邻元素的全局索引。

    //我看懂了 gtid只取奇数block里面的global thread id，不取偶数的
    
    //加载两个元素到shared memory的对应位置
    smen[tid] = d_int[gtid] + d_int[gtid+blockSize];//奇数block的tid+偶数block的tid 这里gtid不会越界的原因是启动的线程数被减半了
    __syncthreads();

    for(int unsigned index = blockDim.x/2; index > 0; index >>= 1){
        if(tid < index){
            smem[tid] += smem[tid+index];
            //首先解决了warp divergence问题，warp的所有线程都进行这个分支，这句话我说的，应该有问题，warp还是只有一半会执行？
            //其次解决了bank conflict问题，sharedmemory的读写是并行的
            //最后解决了空闲线程的问题，让一个线程负责两个sharedmemory的元素
            //第一轮启动一半的warp
            //第二轮只有1/4的warp参加运算
            //第三轮只有1/8的warp参与运算
            
            //比reduce v2可以少启动一半的线程，第一轮的时候所有warp都在运行
        }
        __syncthreads();
    }

    if(tid == 0){
        d_out[blockDim.x] = smem[0];
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
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v3<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v3 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
