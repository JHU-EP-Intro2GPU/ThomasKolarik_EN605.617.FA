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
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  result[thread_idx] = curand(&state[thread_idx]) % MAX;
}
    
// Executes matrix multiplication using the CUDA BLAS library.
// Data is populated with random numbers, which are generated using
// the cuda random number generator library
void executeMultTest(const int totalThreads, const int blockSize, const int numBlocks)
{
    const unsigned int seed = 1;
    
    // Allocate space for all of the device Matricies
    float *hostA = (float*)malloc(totalThreads*sizeof(float));
    float *hostB = (float*)malloc(totalThreads*sizeof(float));
    float *hostC = (float*)malloc(totalThreads*sizeof(float));
    
    cublasStatus status;
    cublasInit();
    
    // Initialize random number generation
    curandState_t* states;
    cudaMalloc((void**) &states, totalThreads * sizeof(curandState_t));
    init<<<numBlocks, blockSize>>>(seed, states);

    // Allocate space for all of the device matricies
    float *gpuA = (float*)cublasAlloc(totalThreads*sizeof(float));
    float *gpuB = (float*)cublasAlloc(totalThreads*sizeof(float));
    float *gpuC = (float*)cublasAlloc(blockSize*blockSize*sizeof(float));

    /* invoke the kernel to get some random numbers */
    populateCudaRandom<<<numBlocks, blockSize>>>(states, gpuA);
    populateCudaRandom<<<blockSize, numBlocks>>>(states, gpuB);
    populateCudaRandom<<<blockSize, blockSize>>>(states, gpuC);

    float* AA; float* BB; float* CC;

    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(HA*WA,sizeof(float),(void**)&AA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(HB*WB,sizeof(float),(void**)&BB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(HC*WC,sizeof(float),(void**)&CC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    /*SET MATRIX*/
    status=cublasSetMatrix(numBlocks,blockSize,sizeof(float),hostA,numBlocks,gpuA,numBlocks);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasSetMatrix(blockSize,numBlocks,sizeof(float),hostB,blockSize,gpuB,blockSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    /*KERNEL*/
    cublasSgemm('n','n',numBlocks,numBlocks,blockSize,1,gpuA,numBlocks,gpuB,blockSize,0,gpuC,numBlocks);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
    cublasGetMatrix(numBlocks,blockSize,sizeof(float),gpuC,blockSize,hostC,blockSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (A)\n");
      return EXIT_FAILURE;
    }
    
    // Free all memory allocations
    free(hostA);
    free(hostB);
    free(hostC);
    cudaFree(states);
    status = cublasFree(gpuA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (A)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(gpuB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (B)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(gpuC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (C)\n");
      return EXIT_FAILURE;
    }
}

// Executes the given test type 
// Data is filled with random numbers.
void executeGPUTest(const int totalThreads, const int blockSize, const int numBlocks, const gpu_tests_enum testType)
{    
    switch(testType)
    {
        case BLAS:
            // Linear Algrebra Multiply Operations
            executeGlobalMathOperations<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, numBlocks * blockSize);
            break;
        case SOLVE:
            // Executes a matrix solver
            executeSharedMathOperations<<<numBlocks, blockSize, 3 * totalThreads * sizeof(int)>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, totalThreads);
            break;
        default:
            std::cout << "Unknown test type " << testType << "!" << std::endl;
            break;
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
    executeGPUTest(totalThreads, blockSize, numBlocks, testType);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Execution took: " << totalTime.count() << " seconds." << std::endl;    
    return 0;
}
