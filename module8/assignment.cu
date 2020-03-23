#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>

// Initializes the random number generate for use inside of a kernal
__global__ void initCudaRandom(unsigned int seed, curandState_t* states)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init(seed,
              thread_idx,
              0,
              &states[thread_idx]);
}

// Populate the given result array with a random number given a state.
__global__ void populateCudaRandom(curandState_t* state, float* result)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    result[thread_idx] = curand(&state[thread_idx]);
}
    
// Executes matrix multiplication using the CUDA BLAS library.
// Data is populated with random numbers, which are generated using
// the cuda random number generator library
void executeMultTest(const int totalThreads, const int blockSize, const int numBlocks)
{
    const unsigned int seed = 1;
    
    // Allocate space for all of the device Matricies
    float *hostC = (float*)malloc(blockSize*blockSize*sizeof(float));
    
    cublasStatus status;
    cublasInit();
    
    // Initialize random number generation
    curandState_t* states;
    cudaMalloc((void**) &states, totalThreads * sizeof(curandState_t));
    initCudaRandom<<<numBlocks, blockSize>>>(seed, states);

    // Allocate space for all of the device matricies
    float *gpuA;
    float *gpuB;
    float *gpuC;

    status=cublasAlloc(totalThreads,sizeof(float),(void**)&gpuA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "Unable to allocate memory for gpuA!\n");
      return;
    }

    status=cublasAlloc(totalThreads,sizeof(float),(void**)&gpuB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "Unable to allocate memory for gpuB!\n");
      return;
    }

    status=cublasAlloc(blockSize*blockSize,sizeof(float),(void**)&gpuC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "Unable to allocate memory for gpuC!\n");
      return;
    }
    
    // Populate the inputs with random numbers
    populateCudaRandom<<<numBlocks, blockSize>>>(states, gpuA);
    populateCudaRandom<<<blockSize, numBlocks>>>(states, gpuB);

    // Do the matrix multiplication
    cublasSgemm('n','n',numBlocks,numBlocks,blockSize,1,gpuA,numBlocks,gpuB,blockSize,0,gpuC,numBlocks);

    // Check for any errors
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "Error during multiplication!\n");
      return;
    }
    
    // Copy the matrix to a host array
    cublasGetMatrix(blockSize,blockSize,sizeof(float),gpuC,blockSize,hostC,blockSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "Error during Matrix Extraction!\n");
      return;
    }
    
    // Free all memory allocations
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

    // Use the cuda blas and random number generator libraries and time how long it takes
    // to execute.
    auto startTime = std::chrono::system_clock::now();
    executeMultTest(totalThreads, blockSize, numBlocks);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Execution took: " << totalTime.count() << " seconds." << std::endl;    
    return 0;
}
