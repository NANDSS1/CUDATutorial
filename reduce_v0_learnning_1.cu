#include<bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

template<int blockSize>//模板，接收一个整形数据，执行的时候填上这个整形数据reduce_kernel<256><<<gridSize, blockSize>>>(d_input, d_output, N);

__global__ void reduce_v0(float* d_in,float* d_out){
    __shared__ float smem[blockSize];//申请这么大的共享内存在一个block里面，属于是block,它用于在线程块内部存储临时数据
    int tid = threadIdx.x;//获取线程在block里面的id
    int gtid = blockIdx.x * blockSize + threadIdx.x;//获取一个全局id

    smem[tid] = d_in[gtid];//把数据依次加载到block的shared memory里面
    __syncthreads();//等待线程同步

    for(int index = 1;index < blockDim.x; index *= 2){//每一层只进行一次循环
        if (tid % (2 * index) == 0){//选择特定的tid才相加
            //printf("index:%d\n",index);
            smem[tid] += smem[tid + index];
            //第一轮按照索引 index = 1 ，只有tid0+tid1 tid2+tid3 tid4+tid5
            //第二轮按照索引 index = 2 ，只有tid0+tid2 tid4+tid6 tid8+tid10
            //第三轮按照索引 index = 4，只有tid0+tid4 tid8+tid12
            //....
            //最后就是tid0+tidblockSize-1
        }
        __syncthreads();//线程同步
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
        //把每个block的tid0的结果写会global memory
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
    cudaSetDevice(0);//选择第一个gpu
    cudaDeviceProp deviceProp;//存储关于 GPU 设备的属性信息
    cudaGetDeviceProperties(&deviceProp, 0);//获取第一个gpu的属性
    const int blockSize = 256;//设定block size
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //(N + 256 - 1) / 256表示对N/256的向上取整，保证所有数据都能分配上thread
    //-1的原因是如果N刚好是256的倍数，分配的block也不会多，如果N=512，不-1，分配的block就是3，如果-1，分配的block就是2

    float *a = (float *)malloc(N * sizeof(float));//申请host上的内存
    float *d_a;
    cudaMalloc((void**)&d_a,N * sizeof(float));//申请device上的内存

    float *out = (float*)malloc((GridSize) * sizeof(float));//申请的float数量和block数量一致
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }//初始化host上的数据


    float groudtruth = N * 1.0f;//累积误差的标准

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);//host cpoy data to device

    dim3 Grid(GridSize);//在x方向申请这么多个block，用作<<<,>>>的第一个参数
    dim3 Block(blockSize);//在x方向申请这么多个thread，用作<<<,>>>的第二个参数

    cudaEvent_t start, stop;//定义两个事件用作定时
    cudaEventCreate(&start);//创建一个开始定时器
    cudaEventCreate(&stop);//创建一个结束定时器
    cudaEventRecord(start);//开始定时器记录一个时间戳

    reduce_v0<blockSize><<<Grid,Block>>>(d_a, d_out);//执行核函数,这里用模板传参的

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);//device copy data to host

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