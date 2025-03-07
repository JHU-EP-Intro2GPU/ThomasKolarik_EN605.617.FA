//Based on the work of Andrew Krepps
#include <iostream>
#include <random>
#include <stdio.h>


__global__ void add(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] + b[thread_idx];
}

__global__ void subtract(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] - b[thread_idx];
}

__global__ void mult(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] * b[thread_idx];
}

__global__ void mod(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] % b[thread_idx];
}

void printArray(const int * const arr, const int xSize, const int ySize)
{
    for (size_t i = 0; i < xSize; ++i)
    {
        for(size_t j = 0; j < ySize; ++j)
        {
            std::cout << arr[i * ySize + j] << " ";
        }
        std::cout << '\n';
    }
    
    std::cout << std::flush;
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = 256;
    int blockSize = 256;
    
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }
    

    int numBlocks = totalThreads/blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks*blockSize;
        
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }
    
    int a[totalThreads], b[totalThreads], c[totalThreads];
    
    int *gpu_a, *gpu_b, *gpu_c;

    cudaMalloc((void**)&gpu_a, totalThreads * sizeof(int));

    cudaMalloc((void**)&gpu_b, totalThreads * sizeof(int));

    cudaMalloc((void**)&gpu_c, totalThreads * sizeof(int));
    
    // Create a random generate that will generate random numbers from 0 to 4.
    // Use a set seed so output is deterministic
    unsigned seed = 12345;
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> dist(0,4);
    
    for (size_t i = 0; i < totalThreads; ++i)
    {
        a[i] = i;
        b[i] = dist(gen);
    }
    
    std::cout << "A:" << std::endl;
    printArray(a, numBlocks, blockSize);
    std::cout << "B:" << std::endl;
    printArray(b, numBlocks, blockSize);
    
    cudaMemcpy(gpu_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    
    // Add all of the numbers c[i] = a[i] + b[i];
    add<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Add: " << std::endl;
    printArray(c, numBlocks, blockSize);
    
    // Subtract all of the numbers c[i] = a[i] - b[i];
    subtract<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sub: " << std::endl;
    printArray(c, numBlocks, blockSize);
    
    // Multiply all of the numbers c[i] = a[i] * b[i];
    mult<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Mult: " << std::endl;
    printArray(c, numBlocks, blockSize);
    
    // Mod all of the numbers c[i] = a[i] % b[i];
    mod<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Mod: " << std::endl;
    printArray(c, numBlocks, blockSize);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    return 0;
}
