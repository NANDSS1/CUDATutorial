#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

#define N 32
__global__ void sum(float* x){
    int block_id = blockIdx.x;//记录block的索引
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;//记录线程的全局索引
    int local_tid = threadIdx.x;//记录线程的block内索引

    //cuda里面执行执行printf
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}//global为cpu启动，gpu执行的的函数声明

int main(){
    int nbytes = N * sizeof(float);//申请N个float
    float *dx,*hx;//申请device的指针和host的指针
    cudaMalloc((void**)&dx,nbytes);//用二级指针给device申请内存
    hx = (float*)malloc(nbytes);
    for(int i = 0;i < N;i++){
        hx[i] = i;
        std::cout<<hx[i]<<std::endl;
    }

    cudaMemcpy(dx,hx,nbytes,cudaMemcpyHostToDevice);//进行数据的copy

    sum<<<1,N>>>(dx);//创建线程，block=1，线程数量=N

    cudaMemcpy(hx,dx,nbytes,cudaMemcpyDeviceToHost);
    
    std::cout<<"hx current:\n"<<std::endl;

    for(int i = 0;i < N;i++){
        std::cout<<hx[i]<<std::endl;
    }

    cudaFree(dx);
    free(hx);

    return 0;

    //初始化hx数据
}
