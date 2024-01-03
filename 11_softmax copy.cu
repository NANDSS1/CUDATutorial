#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>

#define WarpSize 32

bool CheckResult(float *out, float* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
          return false;
      }
    }
    return true;
}

float* softmaxCPU(float* input, float* result, int rows, int cols){
  for (int j = 0; j < rows; j++)
  {
    float total = 0;
    float MAX = 0;
    for(int i = 0; i < cols; i++)
    {
      MAX = max(input[j * cols + i], MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      total += exp(input[j * cols + i] - MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      result[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
    }
  }

  return result;
}
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int warp_width = WarpSize>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = warp_width / 2; mask > 0; mask /= 2) {
    // you can change L61 with __shfl_down_sync like 6_warp_level_reduce and see performance change
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
  //return __expf(x);//fast math
  return exp(x);
}

template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
  return 10000000000;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  //return __fdividef(a, b);//fast math
  return a / b;
}

template<int VecSize>
__device__ void load(const float* src, float* dst, int row, const int row_size, const int col) {
  using VecType = VectorType<float, VecSize>;
  const int offset = (row * row_size + col) / VecSize;
  *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<VecType*>(const_cast<float*>(src)) + offset);
}


template<int VecSize>
__device__ void store(float* dst, float* src, int row, const int row_size, const int col) {
  using VecType = VectorType<float, VecSize>;
  const int offset = (row * row_size + col) / VecSize;
  *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<VecType*>(src);
}


//这个softmax是一个warp求一行的softmax值
template<int pack_size, int cols_per_thread,
         int warp_width, int rows_per_thread>//1, 1024 / 32, 32, 1
/*rows_per_thread和col_per_thread是用于确定每个线程处理的数据块大小的参数。
- rows_per_thread表示每个线程处理的连续行数。在Warp Softmax函数中，每个线程负责加载、计算和存储一定数量的行的数据。
- col_per_thread表示每个线程处理的连续列数。在Warp Softmax函数中，每个线程负责加载、计算和存储一定数量的列的数据。*/
__global__ void WarpSoftmax(const float* src, float* dst, const int rows, const int cols) {
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * warp_width);
  float buf[rows_per_thread][cols_per_thread];//一个warp处理一行数据，分别定义缓存，每个线程的buf是[1][[1024/32]]，这里1024/32，32应该就是一个warp的线程数量，然后一个线程负责处理1024/32这么多列数据，后面的1应该是一列的最大值？
  const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;//计算warp的全局id，warp是纵向的，横向x是线程数量
  const int num_global_warp = gridDim.y * blockDim.y;//计算warp的数量
  const int lane_id = threadIdx.x;//获得每个warp内的线程id，一个block的维度是(32,8) y方向8个warp，每个warp在x方向32个线程
  const int step = num_global_warp * rows_per_thread;//计算步长，防止block里面的warp在y方向小于数据的数量的列数，老操作了
  for (int row = global_warp_id * rows_per_thread; row < rows; row += step) {//一个warp去算一行，这里计算可以计算row，row=当前warp在全局y方向上的id
    float thread_max[rows_per_thread];//thread_max收集每行的最大值

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {//老操作，小马拉大车
      thread_max[row_id] = -Inf<float>();//定义一个很小的数，便于可以被max
      float* row_buf = buf[row_id];//取一行的buf

      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {//vector local sum reduce 把每个线程处理到的数据的的最大值给输出
        const int pack_offset = pack_id * pack_size;//定义向量化的偏移
        const int col = (pack_id * warp_width + lane_id) * pack_size;//定义线程具体处理哪行
        if (col < cols) {
          // load (row+row_id, col) data from src to reg row_buf
          load<pack_size>(src, row_buf + pack_offset, row + row_id, rows, col);//向量化的load

          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);//vector local reduce的时候计算一个max
          }//处理单个vector
        } else {

          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<float>(); }
        }
      }
    }
    float warp_max[rows_per_thread];

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {//进行warp level reduce，把一行的最大值求出来了
      warp_max[row_id] = WarpReduce<MaxOp, float, warp_width>(thread_max[row_id]);
    }
    float thread_sum[rows_per_thread];

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {//thread local sum
      thread_sum[row_id] = 0;
      float* row_buf = buf[row_id];

      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
        thread_sum[row_id] += row_buf[i];
      }
    }
    float warp_sum[rows_per_thread];

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {//warp sum
      warp_sum[row_id] = WarpReduce<SumOp, float, warp_width>(thread_sum[row_id]);
    }

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      float* row_buf = buf[row_id];

      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
      }

      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * warp_width + lane_id) * pack_size;
        if (col < cols) {
          store<pack_size>(dst, row_buf + i * pack_size, row + row_id, rows, col);
        }
      }
    }
  }
}

int main(){
    float milliseconds = 0;
    const int N = 1000 * 1024;
    float *src = (float *)malloc(N * sizeof(float));
    float *d_src;
    cudaMalloc((void **)&d_src, N * sizeof(float));

    //int gridSize = ;//2d block, blockx=32,blocky=num warps in a block,griddimy=block nums
    //int blockSize = 256;
    float *dst = (float*)malloc(N * sizeof(float));
    float *d_dst;
    cudaMalloc((void **)&d_dst, N * sizeof(float));
    float *groudtruth = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        src[i] = 1;
    }

    groudtruth = softmaxCPU(src, dst, 1000, 1024);

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(1, 125);//y轴125个block,
    dim3 Block(32, 8);//x轴32个threads组成一个warp访问一行,y轴8个threads,8*125=1000行,每个warp处理一行
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    WarpSoftmax<1, 1024 / 32, 32, 1><<<Grid, Block>>>(d_src, d_dst, 1000, 1024);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(dst, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",dst[i]);
        }
        printf("\n");
    }
    printf("WarpSoftmax latency = %f ms\n", milliseconds);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(src);
    free(dst);
    free(groudtruth);
}
