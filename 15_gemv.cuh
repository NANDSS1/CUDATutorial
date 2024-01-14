#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string>
#include <stdexcept>
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

/*这是一个宏定义的代码片段，用于在CUDA编程中检查CUDA函数调用是否成功。它的作用是：

- 定义了一个宏`CHECK(call)`，它接受一个CUDA函数调用作为参数。
- 在宏内部，使用`const cudaError_t error_code = call;`执行CUDA函数调用，并将返回值保存在`error_code`变量中。
- 如果`error_code`不等于`cudaSuccess`，即CUDA函数调用失败，那么会输出错误信息，并通过`exit(1)`终止程序。
- 错误信息包括文件名、行号、错误代码和错误文本。

这个宏的目的是简化CUDA函数调用后的错误检查过程，使得代码更加简洁和易读。在需要进行CUDA函数调用的地方，可以使用`CHECK()`宏来代替繁琐的错误检查代码。例如：

```cpp
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
CHECK(cudaGetLastError());
```

上述代码片段中，`cudaMemcpy()`函数调用后紧跟着`CHECK()`宏，可以快速检查`cudaMemcpy()`是否成功执行。如果发生错误，宏会输出相应的错误信息并终止程序。*/
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template<typename T>
struct Vec {
    static constexpr int size = 4;//一次load的数据量
};
template<>
struct Vec<half> {
    static constexpr int size = 8;
};

/*`__device__`和`__forceinline__`是CUDA中的两个标识符，用于修饰函数或变量的行为和属性。

1. `__device__`：
   - `__device__`用于修饰函数或变量，指示它们在设备（GPU）上执行。修饰的函数称为`__device__`函数，修饰的变量称为`__device__`变量。
   - `__device__`函数可以从主机端调用，也可以从设备端调用。在设备端调用时，`__device__`函数在设备上执行，并且只能访问设备上的内存。
   - `__device__`变量是全局变量，存储在设备端的全局内存中，可以被所有设备线程访问。

2. `__forceinline__`：
   - `__forceinline__`用于修饰函数，强制启用内联展开优化。内联展开是将函数的代码插入到调用处，避免了函数调用的开销。
   - 使用`__forceinline__`修饰的函数会被编译器尽可能地进行内联展开，但编译器有权选择是否进行内联展开。
   - `__forceinline__`在某些情况下可以提高性能，但也可能导致代码膨胀和编译时间增加。因此，应谨慎使用，并在实际测试中评估其效果。

综上所述，`__device__`和`__forceinline__`是CUDA中的两个修饰符，用于指示函数或变量在设备上执行以及启用内联展开优化。这些修饰符可以提高CUDA代码的性能和效率。*/
template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<>
struct SumOp<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hadd(a, b); }
};

template<template<typename> class ReductionOp, typename T>//ReductionOp可以换成SumOp
__device__ __forceinline__ T warpReduce(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
        /*具体地，函数使用了一个循环来实现归约操作。循环中，通过不断将掩码（mask）右移一位，将当前线程的值与距离当前线程一定距离的其他线程的值进行运算，并使用指定的归约操作对这两个值进行处理。每次循环后，将得到的新的值作为下一次迭代的输入。经过多次迭代后，最终得到线程束内的归约结果。

这个函数的目的是为了提高代码的效率，通过在线程束内部进行归约操作，减少数据传输和计算的开销，从而加快程序的执行速度。*/
    }
    return val;
}
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;//第几个warp
    int lane_id = tid % 32;//warp内第几个线程
    int warp_nums = (blockDim.x + 31) / 32;//比如是block的维度是33-63 计算出来warp_nums都是2，这是为了向上取整
    /*在函数示例中，通过将线程ID除以32得到它所属的warp ID，然后计算出当前block中有多少个warp。
具体而言，blockDim.x表示当前block中有多少个线程，(blockDim.x + 31) / 32表示当前block中有多少个warp。
这个计算式的含义是，将blockDim.x加上31，然后除以32，可以将其向上取整到最接近的32的倍数，从而确保每个warp都有32个线程。
这个计算式的结果通常会被用作其他CUDA函数的参数，例如__syncthreads()，以确保每个warp内的线程都能够正确同步。*/
    static __shared__ float warpres[64];//分配64个float的寄存器
    val = warpReduce<ReductionOp, T>(val);//在一个warp里面做reduce
    if (lane_id == 0){
        warpres[warp_id] = val;//将每个warp计算出的值放到对应的寄存器里面
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warpres[tid] : 0;//选择warp_nums前tid
    return warpReduce<ReductionOp, T>(warp_val);
}

// 一个block处理一行
// mat * vec = {M, N} * {N, 1}/{1, N}
template<int VECS_PER_THREAD, int VEC_SIZE>//两个模板，第一个是每个线程处理的数量，第二个是vector的大小
__global__ void gemv(float* matrix, float* vector, float* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float thread_local_sum = 0.0f;//每个线程定义一个寄存器
    for(int i = 0; i < VECS_PER_THREAD; i++) {//VECS_PER_THREAD是每个向量需要处理的向量的个数
        float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE]); // 4 * half2 //bid*cols定位具体在mat的哪一行，然后tid*VECSIZE就可以计算thread向量化操作的起始地址
        float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE]);//每个thread起始地址之间查了VEC_SIZE，VEC_SIZE是向量的个数。一次可以load 128bit的数据，就是4个fp32和8个half
        thread_local_sum += mat4[i].x * vec4[i].x;
        thread_local_sum += mat4[i].y * vec4[i].y;
        thread_local_sum += mat4[i].z * vec4[i].z;
        thread_local_sum += mat4[i].w * vec4[i].w;//向量化的load，load float4，向量化的访存，但是计算的时候要用标量
    }
    //reduce to get the final val
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);//计算一个block内的reduce
    //store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

struct half8 {
    half2 x;
    half2 y;
    half2 w;
    half2 z;
};

template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {//用安培及安培以上架构来做half2
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //float thread_local_sum = 0.0f;
    half thread_local_sum = 0;
    for(int i = 0; i < VECS_PER_THREAD; i++) {//cols就是列数N，block id就代表行数，一个block处理一行
        float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE]); // 4 * half2
        float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE]);
        half2* vec_h1 = (half2*)&vec4[i].x;
        half2* vec_h2 = (half2*)&vec4[i].y;
        half2* vec_h3 = (half2*)&vec4[i].z;
        half2* vec_h4 = (half2*)&vec4[i].w;
        half2* mat_h1 = (half2*)&mat4[i].x;
        half2* mat_h2 = (half2*)&mat4[i].y;
        half2* mat_h3 = (half2*)&mat4[i].z;
        half2* mat_h4 = (half2*)&mat4[i].w;   
        half2 res1 = __hmul2(*mat_h1, *vec_h1);
        half2 res2 = __hmul2(*mat_h2, *vec_h2);
        half2 res3 = __hmul2(*mat_h3, *vec_h3);
        half2 res4 = __hmul2(*mat_h4, *vec_h4); 
        half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        thread_local_sum = __hadd(res.x, res.y);
        // float2 res1 = __half22float2(__hmul2(*mat_h1, *vec_h1));
        // float2 res2 = __half22float2(__hmul2(*mat_h2, *vec_h2));
        // float2 res3 = __half22float2(__hmul2(*mat_h3, *vec_h3));
        // float2 res4 = __half22float2(__hmul2(*mat_h4, *vec_h4));
        // thread_local_sum += res1.x;
        // thread_local_sum += res1.y;
        // thread_local_sum += res2.x;
        // thread_local_sum += res2.y;
        // thread_local_sum += res3.x;
        // thread_local_sum += res3.y;
        // thread_local_sum += res4.x;
        // thread_local_sum += res4.y;
        if(i == 0 && tid == 0 && bid == 0) {
            printf("thread sum = %f\n", (float)thread_local_sum); // 8
            // printf("res1.x = %f\n", res1.x); // 1
            // printf("res1.y = %f\n", res1.y);
        }
    }
    //reduce to get the final val
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    // float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        printf("block reduce_res = %f\n", (float)reduce_res);
        // res[blockIdx.x] = __float2half(reduce_res);
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}


template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(M);//多少行就分配多少个block
        dim3 Block(THREAD_NUMS);//每个block分配多少个thread
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
