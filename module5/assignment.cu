//Based on the work of Andrew Krepps
#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>

// Device GPU add c[i] = a[i] + b[i]
__device__ void add(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] + b[thread_idx];
}

// Device GPU subtract c[i] = a[i] - b[i]
__device__ void subtract(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] - b[thread_idx];
}

// Device GPU multiply c[i] = a[i] * b[i]
__device__ void mult(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] * b[thread_idx];
}

// Device GPU div c[i] = a[i] / b[i]
__device__ void div(int *a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] / b[thread_idx];
}

// Device GPU mod c[i] = a[i] % b[i]
__device__ void mod(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] % b[thread_idx];
}

// Global GPU add c[i] = a[i] + b[i]
__global__ void addGlobal(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] + b[thread_idx];
}

// Global GPU subtract c[i] = a[i] - b[i]
__global__ void subtractGlobal(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] - b[thread_idx];
}

// Global GPU multiply c[i] = a[i] * b[i]
__global__ void multGlobal(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] * b[thread_idx];
}

// Global GPU div c[i] = a[i] / b[i]
__global__ void divGlobal(int *a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] / b[thread_idx];
}

// Global GPU mod c[i] = a[i] % b[i]
__global__ void modGlobal(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] % b[thread_idx];
}

// Copies data from one array to another
__device__ void copyData(const int * const srcArr,
						 int * const destArr,
						 const int tid,
						 const int size)
{
	// Copy data into temp store
	for(int i = 0; i<size; i++)
	{
		destArr[i+tid] = srcArr[i+tid];
	}
	__syncthreads();
}

__global__ void executeSharedMathOperations(int * a, int * b, int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int sharedA[];
    extern __shared__ int sharedB[];
    extern __shared__ int sharedRet[];
    
    copyData(a, sharedA, tid, size);
    copyData(b, sharedB, tid, size);
    
    // Add sharedA to sharedB and store in addDest
    add(sharedA, sharedB, sharedRet);
    copyData(sharedRet, addDest, tid, size);
    
    // Subtract sharedB from sharedA and store in subDest
    subtract(sharedA, sharedB, sharedRet);
    copyData(sharedRet, subDest, tid, size);
    
    // Multiply sharedA to sharedB and store in mutlDest
    mult(sharedA, sharedB, sharedRet);
    copyData(sharedRet, multDest, tid, size);
    
    // Divide sharedA by sharedB and store in divDest
    div(sharedA, sharedB, sharedRet);
    copyData(sharedRet, divDest, tid, size);
    
    // Mod sharedA by sharedB and store in modDest
    mod(sharedA, sharedB, sharedRet);
    copyData(sharedRet, modDest, tid, size);
}

__global__ void executeGlobalMathOperations(int * a, int * b, int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int sharedA[size];
    __shared__ int sharedB[size];
    __shared__ int sharedRet[size];
    
    copyData(a, sharedA, tid, size);
    copyData(b, sharedB, tid, size);
    
    // Add sharedA to sharedB and store in addDest
    addGlobal(sharedA, sharedB, sharedRet);
    copyData(sharedRet, addDest, tid, size);
    
    // Subtract sharedB from sharedA and store in subDest
    subGlobal(sharedA, sharedB, sharedRet);
    copyData(sharedRet, subDest, tid, size);
    
    // Multiply sharedA to sharedB and store in mutlDest
    multGlobal(sharedA, sharedB, sharedRet);
    copyData(sharedRet, multDest, tid, size);
    
    // Divide sharedA by sharedB and store in divDest
    divGlobal(sharedA, sharedB, sharedRet);
    copyData(sharedRet, divDest, tid, size);
    
    // Mod sharedA by sharedB and store in modDest
    modGlobal(sharedA, sharedB, sharedRet);
    copyData(sharedRet, modDest, tid, size);
}

// Host (Cpu) add c[i] = a[i] + b[i]
void hostAdd(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

// Host (Cpu) sub c[i] = a[i] - b[i]
void hostSub(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

// Host (Cpu) multiply c[i] = a[i] * b[i]
void hostMult(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

// Host (Cpu) divide c[i] = a[i] / b[i]
void hostDiv(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (b[i] != 0)
        {
            c[i] = a[i] / b[i];
        }
        else
        {
            c[i] = 0;
        }
    }
}

// Host (Cpu) mod c[i] = a[i] % b[i]
void hostMod(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        // Protect against divide by 0. 
        // cuda code catches this error and sets result to 0 by default.
        if (b[i] == 0)
        {
            c[i] = 0;
        }
        else
        {
            c[i] = a[i] % b[i];
        }
    }
}

// Executes each of the host (cpu) tests by creating local memory and executing all 5 math operations on the data.
// The data is filled with random numbers that uses the same seed as the GPU tests.
void executeHostTest(const int totalThreads, const int blockSize, const int numBlocks)
{
    int a[totalThreads], b[totalThreads], c[totalThreads];
    
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
    
    // Add all of the numbers c[i] = a[i] + b[i];
    hostAdd(a,b,c, totalThreads);
    
    // Subtract all of the numbers c[i] = a[i] - b[i];
    hostSub(a,b,c, totalThreads);
    
    // Multiply all of the numbers c[i] = a[i] * b[i];
    hostMult(a,b,c, totalThreads);
    
    // Divides all of the numbers c[i] = a[i] / b[i]; if b[i] == 0, c[i] = 0
    hostDiv(a,b,c, totalThreads);
    
    // Mod all of the numbers c[i] = a[i] % b[i];
    hostMod(a,b,c, totalThreads);
}

// Executes each of the global memory gpu tests by creating local memory, copying it global memory, and then performing
//  all 5 math operations on the data. The data is filled with random numbers that uses the same seed as the CPU tests.
void executeGlobalTest(const int totalThreads, const int blockSize, const int numBlocks)
{
    int a[totalThreads], b[totalThreads], add_dest[totalThreads], sub_dest[totalThreads], mult_dest[totalThreads], div_dest[totalThreads], mod_dest[totalThreads];
    
    int *gpu_a, *gpu_b, *gpu_add_dest, *gpu_sub_dest, *gpu_mult_dest, *gpu_div_dest, *gpu_mod_dest;

    cudaMalloc((void**)&gpu_a,         totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_b,         totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_add_dest,  totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_sub_dest,  totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_mult_dest, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_div_dest,  totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_mod_dest,  totalThreads * sizeof(int));
    
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
    
    cudaMemcpy(gpu_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    
    executeGlobalMathOperations<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, numBlocks * blockSize);
    
    cudaMemcpy(add_dest, gpu_add_dest,   totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_dest, gpu_sub_dest,   totalThreads*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(mult_dest, gpu_mult_dest, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(div_dest, gpu_div_dest,   totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mod_dest, gpu_mod_dest,   totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_add_dest);
    cudaFree(gpu_sub_dest);
    cudaFree(gpu_mult_dest);
    cudaFree(gpu_div_dest);
    cudaFree(gpu_mod_dest);
}

// Executes each of the shared memory gpu tests by creating local memory, copying it global memory, and then performing
// all 5 math operations on the data. The data is filled with random numbers that uses the same seed as the CPU tests.
void executeSharedTest(const int totalThreads, const int blockSize, const int numBlocks)
{
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
    
    cudaMemcpy(gpu_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    
    // Add all of the numbers c[i] = a[i] + b[i];
    add<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Subtract all of the numbers c[i] = a[i] - b[i];
    subtract<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Multiply all of the numbers c[i] = a[i] * b[i];
    mult<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Divide all of the numbers c[i] = a[i] / b[i];
    div<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Mod all of the numbers c[i] = a[i] % b[i];
    mod<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);    
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
    
    auto startTime = std::chrono::system_clock::now();
    executeHostTest(totalThreads, blockSize, numBlocks);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Host execution took: " << totalTime.count() << " seconds." << std::endl;
    
    startTime = std::chrono::system_clock::now();
    executeGlobalTest(totalThreads, blockSize, numBlocks);
    endTime = std::chrono::system_clock::now();
    totalTime = endTime-startTime;
    std::cout << "Global Memory execution took: " << totalTime.count() << " seconds." << std::endl;
    
    startTime = std::chrono::system_clock::now();
    executeSharedTest(totalThreads, blockSize, numBlocks);
    endTime = std::chrono::system_clock::now();
    totalTime = endTime-startTime;
    std::cout << "Shared Memory execution took: " << totalTime.count() << " seconds." << std::endl;
    
    return 0;
}
