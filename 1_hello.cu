#include <stdio.h>
__global__ void helloFromGPU(){

printf("Hello World from GPU\n");

printf("blockDim:x=%d,y=%d,z=%d,gridDim:x=%d,y=%d,z=%d Current threadIdx=%d,y=%d,z=%d\n",blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z,threadIdx.x,threadIdx.y,threadIdx.z);
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU\n");

    dim3 grid;
    grid.x = 2;
    grid.y = 2;
    dim3 block;
    block.x = 2;
    block.y = 2;
    helloFromGPU<<<grid,block>>>();
    cudaDeviceReset();
    return 0;
}
//sudo apt-get -o Acquire::http::proxy="http://127.0.0.1:7890/" install gcc-7 g++-7 -y