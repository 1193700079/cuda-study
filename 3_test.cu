#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
__global__ void generate_random_numbers(float *numbers, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        numbers[idx] = curand_uniform(&state);
    }
}

int main()
{
    int n = 10;
    float *numbers;
    cudaMalloc(&numbers, n * sizeof(float));

    generate_random_numbers<<<1, n>>>(numbers, n);

    float *host_numbers = new float[n];
    cudaMemcpy(host_numbers, numbers, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        // printf("%f\n", host_numbers[i]);
        std::cout<<host_numbers[i]<<std::endl;
    }

    delete[] host_numbers;
    cudaFree(numbers);

    return 0;
}