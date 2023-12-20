#include<bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FLOAT float

__global__ void vec_add(float* x,float* y,float* z,int N){
    //2d grid 网格
    //计算线程threand index的时候需要先计算block index
    //blockIdx.x表示block在当前层x方向上的索引
    //blockIdx.y * gridDim.x表示当前block上面所有层的block数量
    int idx = blockDim.x * (blockIdx.x+ blockIdx.y * gridDim.x) + threadIdx.x;
    if(idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(float* x,float* y,float* z,int N){
    for(int i = 0;i < N;i++){
        z[i] = x[i] + y[i];
    }
}


int main(){
    int N = 10000;
    int nbytes = N * sizeof(float);


    int bs = 256;//设置block的size

    int s = ceil(sqrt((N+bs-1.0)/bs));
    
    dim3 grid(s,s);//创建一个2d的grid

    //1D的grid就是 int s = ceil((N + bs - 1.) / bs); dim3 grid(s);
    //ceil表示向上取整数 计算出来的结果是(10000+256-1)/256 结果是40.005 
    //40.05向上取整就是41，最多可以存储10496个线程
    //向下取整就是40 10240，其实也可以存储这么多

    //定义device和host上面的指针
    float *dx,*hx;
    float *dy,*hy;
    float *dz,*hz;

    //在device上面申请内存
    cudaMalloc((void**)&dx,nbytes);
    cudaMalloc((void**)&dy,nbytes);
    cudaMalloc((void**)&dz,nbytes);

    //初始化时间
    float milliseconds = 0;

    //在host上面申请内存
    hx = (float*)malloc(nbytes);
    hy = (float*)malloc(nbytes);
    hz = (float*)malloc(nbytes);

    //在host上初始化数据
    for(int i = 0;i < N;i++){
        hx[i] = 1;
        hx[i] = 1;
    }

    //host copy data to device
    cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,nbytes,cudaMemcpyHostToDevice);


    //创建cuda event计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //启动kernel函数
    vec_add<<<grid,bs>>>(dx,dy,dz,N);
    //等待kernel执行完毕

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);


    //cpu计算
    float *hz_cpu_res = (float*)malloc(nbytes);
    vec_add_cpu(hx,hy,hz_cpu_res,N);

    //计算误差
    for(int i = 0;i < N;i++){
        if(fabs(hz_cpu_res[i] - hz[i]) > 1e-6){
            std::cout<<"Result verification failed at element index"<<i<<std::endl;
        }
    }

    std::cout<<"Result right"<<std::endl;

    //带宽的计算，bytes/s 最后除以10的6次方，换算成Gbytes/s,计算的运算的时间
    std::cout<<"Men BW = "<<(float)nbytes*4/milliseconds/1e6<<"(GB/sec)"<<std::endl;

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;





}