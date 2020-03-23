#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>

/* this GPU kernel function is used to initialize the random states */
__global__ void initCudaRandom(unsigned int seed, curandState_t* states)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init(seed,
              thread_idx,
              0,
              &states[blockIdx.x]);
}

__global__ void populateCudaRandom(curandState_t* state, int* result)
{
    const int MAX_RAND = 100;
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    result[thread_idx] = curand(&state[thread_idx]) % MAX_RAND;
}
    
// Executes matrix multiplication using the CUDA BLAS library.
// Data is populated with random numbers, which are generated using
// the cuda random number generator library
void executeMultTest(const int totalThreads, const int blockSize, const int numBlocks)
{
    const unsigned int seed = 1;
    
    // Allocate space for all of the device Matricies
    int *hostA = (int*)malloc(totalThreads*sizeof(int));
    int *hostB = (int*)malloc(totalThreads*sizeof(int));
    int *hostC = (int*)malloc(totalThreads*sizeof(int));
    
    cublasStatus status;
    cublasInit();
    
    // Initialize random number generation
    curandState_t* states;
    cudaMalloc((void**) &states, totalThreads * sizeof(curandState_t));
    initCudaRandom<<<numBlocks, blockSize>>>(seed, states);

    // Allocate space for all of the device matricies
    int *gpuA = (int*)cublasAlloc(totalThreads*sizeof(int));
    int *gpuB = (int*)cublasAlloc(totalThreads*sizeof(int));
    int *gpuC = (int*)cublasAlloc(blockSize*blockSize*sizeof(int));

    /* invoke the kernel to get some random numbers */
    populateCudaRandom<<<numBlocks, blockSize>>>(states, gpuA);
    populateCudaRandom<<<blockSize, numBlocks>>>(states, gpuB);
    populateCudaRandom<<<blockSize, blockSize>>>(states, gpuC);

    /*SET MATRIX*/
    status=cublasSetMatrix(numBlocks,blockSize,sizeof(float),hostA,numBlocks,gpuA,numBlocks);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return;
    }

    status=cublasSetMatrix(blockSize,numBlocks,sizeof(float),hostB,blockSize,gpuB,blockSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return;
    }

    /*KERNEL*/
    cublasSgemm('n','n',numBlocks,numBlocks,blockSize,1,gpuA,numBlocks,gpuB,blockSize,0,gpuC,numBlocks);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return;
    }
    cublasGetMatrix(numBlocks,blockSize,sizeof(float),gpuC,blockSize,hostC,blockSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (A)\n");
      return;
    }
    
    // Free all memory allocations
    free(hostA);
    free(hostB);
    free(hostC);
    cudaFree(states);
    status = cublasFree(gpuA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (A)\n");
      return;
    }
    status = cublasFree(gpuB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (B)\n");
      return;
    }
    status = cublasFree(gpuC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (C)\n");
      return;
    }
}

// Prints an array. Used for debugging
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

    auto startTime = std::chrono::system_clock::now();
    executeMultTest(totalThreads, blockSize, numBlocks);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Execution took: " << totalTime.count() << " seconds." << std::endl;    
    return 0;
}
