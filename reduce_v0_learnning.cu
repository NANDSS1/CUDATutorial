#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//latency: 3.835ms
template<int blockSize>
__global__ void reduce_v0(float *d_in,float *d_out){
    __shared__ float smem[blockSize];//每个block独立使用一块sharedmemory，block里面的每个thread都能看到sharememory里面的数据


    int tid = threadIdx.x;//获取线程的id
    int gtid = blockIdx.x * blockSize + threadIdx.x;//获取全局的id
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();//必须对所有的线程进行同步，这些线程对sharedmemory进行读写之后回到同一起跑线上

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for(int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) {
            smem[tid] += smem[tid + index];//如果对于第一个数，它第一次加的是2，第二次是4，第三次是8....
        }
        __syncthreads();//线程同步
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
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
    const int blockSize = 256;//一个block只处理256个数据
    /*这个表达式确保当 N 不是 256 的整数倍时，也能够包含所有的元素。
    例如，如果 N 是 400，那么 (400 + 256 - 1) / 256 将计算为 2，即需要两个 block。*/
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

    float groudtruth = N * 1.0f;//累积误差的标准

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<blockSize><<<Grid,Block>>>(d_a, d_out);

    /**/
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
