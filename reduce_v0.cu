#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//latency: 3.835ms
/*
template<int blockSize>
__global__ void reduce_v0(float *d_in,float *d_out){
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for(int index = 1; index < blockDim.x; index *= 2) {//分治
        if (tid % (2 * index) == 0) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}*/

/*核心思想就是树形分治，主要是要注意sharedmemory的使用*/
template<int blockSize>
__global__ void reduce_v0(float* d_in,float* d_out){
    __shared__ float smen[blockSize];//定义blockSize那么大的sharedmemory

    int tid = threadIdx.x;//定义一个局部线程id
    int gtid = blockIdx.x * blockSize + threadIdx.x;//定义一个全局线程id gtid = global thread id
    smem[tid] = d_in[gtid];//第一个block的smen加载完了就加载第二个的，依次类推，smem用tid进行查找对应内存，d_in用gtid查找全局内存
    __syncthreads();//等待所有线程同步，有些线程快，有些线程慢

    //并行计算
    for(int index = 1;index < blockDim.x;index *= 2){//分治算法
        if(tid % (2*index) == 0){
            smen[tid] += smen[tid+index];//0-1 2-3 4-5 6-7 etc //0-2 4-6 8-10 etc 最后加的index <= blockDimx.x
        }
        __syncthreads();//每次计算都要等线程同步
    }

    if(tid == 0){//这里要用block内的id
        d_out[blockIdx.x] = smen[0];//每个block sharedmemory的第一位放到d_out里面，d_out的大小就是block的数量
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
    reduce_v0<blockSize><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
