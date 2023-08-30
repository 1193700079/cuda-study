#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. A 2D thread block and 2D grid are used. sumArraysOnHost sequentially
 * iterates through vector elements on the host.
 */

void initialData(int *ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (int)(rand() & 0xFF);
    }

    return;
}

void sumMatrixOnHost(int *A, int *B, int *C, const int nx,
                     const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = 0;
            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

__constant__ float factor; // 一定要是全局 常量

__global__ void constantMemory()
{

    printf("get constant memory:%.2f\n", factor);
}

__device__ float global_factor = 3.2; // GPU 全局静态变量

__global__ void globalMemory(float *out)
{

    printf("device memory:%.2f\n", global_factor);
    *out = global_factor;
}

__global__ void globalMemory2()
{

    printf("device memory:%.2f\n", global_factor);
    global_factor += 1.3;
}

__managed__ float y = 7.0f;

__global__ void unifiedMemory(float *A)
{

    *A += y;
    printf("GPU unified memory:%.2f\n", *A);
}

__global__ void pageLockedMemory(float *A)
{

    printf("GPU page-locked memory:%.2f\n", *A);
}

__global__ void zerocopyMemory(float *A)
{

    printf("GPU zero-copy memory:%.2f\n", *A);
}

__global__ void kernel_1()
{
    __shared__ float k1_shared; //只能在相同的block中访问！
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && id == 0)
    {
        k1_shared = 5.0;
    }
    // if (blockIdx.x == 1 && id == 16)
    // {
    //     k1_shared = 6.0;
    // }
    __syncthreads();
    printf("access local shared in kernel_1,k1_shared=%.2f,blockIdx=%d,threadIdx=%d,threadId=%d\n",
           k1_shared, blockIdx.x, threadIdx.x, id);
}

__shared__ float g_shared;
__global__ void kernel_2()
{
    g_shared = 0.0;
    printf("access global shared in kernel_2,g_shared=%.2f\n", g_shared);
}

// grid 2D block 2D
__global__ void __launch_bounds__(256, 4) sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,
                                                           int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 13;
    int ny = 1 << 13;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 1024;
    int dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
           grid.y,
           block.x, block.y, iElaps);
    // check kernel error
    // CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    // checkResult(hostRef, gpuRef, nxy);

    // 常量的时候
    float h_factor = 2.3;
    CHECK(cudaMemcpyToSymbol(factor, &h_factor, sizeof(float), 0, cudaMemcpyHostToDevice)); // 设置常量
    constantMemory<<<1, 8>>>();
    CHECK(cudaDeviceSynchronize());

    float h_a;  // 一个变量   一定要这么写
    float *d_a; // 一个指针
    CHECK(cudaMalloc((void **)&d_a, sizeof(float)));
    globalMemory<<<1, 8>>>(d_a);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h_a, d_a, sizeof(float), cudaMemcpyDeviceToHost)); // 拿到返回值
    printf("host memory:%.2f\n", h_a);

    float h_x = 10.8;
    CHECK(cudaMemcpyToSymbol(global_factor, &h_x, sizeof(float), 0, cudaMemcpyHostToDevice)); // 设置常量
    globalMemory2<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());                                                             // 保证计算完成
    CHECK(cudaMemcpyFromSymbol(&h_x, global_factor, sizeof(float), 0, cudaMemcpyDeviceToHost)); // 拿到返回值 写法1
    printf("cudaMemcpyFromSymbol result is:%.2f\n", h_x);

    float *pd_x; // 指向gpu的地址
    CHECK(cudaGetSymbolAddress((void **)&pd_x, global_factor));
    CHECK(cudaMemcpy(&h_x, pd_x, sizeof(float), cudaMemcpyDeviceToHost)); // 拿到返回值 写法2
    printf("cudaGetSymbolAddress result is:%.2f\n", h_x);

    // 采用统一虚拟内存 UVA
    float *d_mem = NULL;
    CHECK(cudaMalloc((void **)&d_mem, sizeof(float)));
    cudaPointerAttributes pt_Attribute;
    CHECK(cudaPointerGetAttributes(&pt_Attribute, d_mem));
    printf("pointer Attribute:device=%d,devicePointer=%p,type=%d\n",
           pt_Attribute.device, pt_Attribute.devicePointer, pt_Attribute.type);
    CHECK(cudaFree(d_mem));

    // 判断gpu是否支持统一内存空间

    int supportManagedMemory = 0;
    CHECK(cudaDeviceGetAttribute(&supportManagedMemory, cudaDevAttrManagedMemory, dev));
    if (0 == supportManagedMemory)
    {
        printf("allocate managed memory is not supported\n");
        return -1;
    }

    printf("unified memory model is supported\n");

    float *unified_mem = NULL;
    CHECK(cudaMallocManaged((void **)&unified_mem, sizeof(float)));
    *unified_mem = 5.7;
    unifiedMemory<<<1, 2>>>(unified_mem);
    CHECK(cudaDeviceSynchronize());
    printf("CPU unified memory:%.2f\n", *unified_mem);
    CHECK(cudaFree(unified_mem));

    // 页锁定内存
    float *h_PinnedMem = NULL;
    CHECK(cudaMallocHost((float **)&h_PinnedMem, sizeof(float)));
    *h_PinnedMem = 4.8;
    printf("CPU page-locked memory:%.2f\n", *h_PinnedMem);
    pageLockedMemory<<<1, 1>>>(h_PinnedMem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFreeHost(h_PinnedMem));

    // 零拷贝内存
    float *h_zerocpyMem = NULL;
    CHECK(cudaHostAlloc((float **)&h_zerocpyMem, sizeof(float), cudaHostAllocDefault));
    *h_zerocpyMem = 4.5;
    printf("CPU zero-copy memory:%.2f\n", *h_zerocpyMem);
    zerocopyMemory<<<1, 1>>>(h_zerocpyMem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFreeHost(h_zerocpyMem));

    // 开启L1缓存  nvcc -Xptxas -dlcm=cg sumMatrixOnGPU-2D-grid-2D-block-integer.cu -o sumMatrixOnGPU-2D-grid-2D-block-integer
    if (deviceProp.globalL1CacheSupported)
    {
        printf("Global L1 cache is supported!\n");
    }
    else
    {
        printf("Global L1 cache is not supported!\n");
    }


    // 共享内存的使用
    kernel_1<<<2, 16>>>();
    printf("======================\n");
    kernel_2<<<2, 16>>>();
    

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
