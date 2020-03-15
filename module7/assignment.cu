//Based on the work of Andrew Krepps
#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>

__constant__ static const int VAL_A = 1;
__constant__ static const int VAL_B = 3;

enum gpu_tests_enum
{
    GLOBAL,
    SHARED,
    CONSTANT,
    REGISTER,
    STREAM,
    NUM_GPU_TESTS
};

gpu_tests_enum& operator++(gpu_tests_enum& e)
{
    return e = (e == NUM_GPU_TESTS) ? GLOBAL : static_cast<gpu_tests_enum>(static_cast<int>(e)+1);
}

std::string gpu_tests_strings[NUM_GPU_TESTS] = {
    "Global",
    "Shared",
    "Constant",
    "Register",
    "Stream"};

// Global GPU add c[i] = a[i] + b[i]
__global__ void addGlob(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] + b[thread_idx];
}

// Global GPU subtract c[i] = a[i] - b[i]
__global__ void subtractGlob(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] - b[thread_idx];
}

// Global GPU multiply c[i] = a[i] * b[i]
__global__ void multGlob(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] * b[thread_idx];
}

// Global GPU div c[i] = a[i] / b[i]
__global__ void divGlob(int *a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] / b[thread_idx];
}

// Global GPU mod c[i] = a[i] % b[i]
__global__ void modGlob(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    c[thread_idx] = a[thread_idx] % b[thread_idx];
}

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

// Device GPU add in register c[i] = a[i] + b[i]
__device__ void addReg(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tempA = a[thread_idx];
    int tempB = b[thread_idx];
    int tempResult = tempA + tempB;
    c[thread_idx] = tempResult;
}

// Device GPU subtract in register c[i] = a[i] - b[i]
__device__ void subtractReg(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tempA = a[thread_idx];
    int tempB = b[thread_idx];
    int tempResult = tempA - tempB;
    c[thread_idx] = tempResult;
}

// Device GPU multiply in register c[i] = a[i] * b[i]
__device__ void multReg(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tempA = a[thread_idx];
    int tempB = b[thread_idx];
    int tempResult = tempA * tempB;
    c[thread_idx] = tempResult;
}

// Device GPU div in register c[i] = a[i] / b[i]
__device__ void divReg(int *a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tempA = a[thread_idx];
    int tempB = b[thread_idx];
    int tempResult = tempA / tempB;
    c[thread_idx] = tempResult;
}

// Device GPU mod in register c[i] = a[i] % b[i]
__device__ void modReg(int * a, int * b, int * c)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tempA = a[thread_idx];
    int tempB = b[thread_idx];
    int tempResult = tempA % tempB;
    c[thread_idx] = tempResult;
}

// Executes all 5 shared math operations
__global__ void executeSharedMathOperations(int * a, int * b, int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int sharedMem[];
    
    // Use offsets in the shared mem to create arrays.
    int * sharedA = &sharedMem[0];
    int * sharedB = &sharedMem[size];
    int * sharedRet = &sharedMem[2*size];
    
    sharedA[tid] = a[tid];
    sharedB[tid] = b[tid];
    
    // Add sharedA to sharedB and store in addDest
    add(sharedA, sharedB, sharedRet);
    addDest[tid] = sharedRet[tid];
    
    // Subtract sharedB from sharedA and store in subDest
    subtract(sharedA, sharedB, sharedRet);
    subDest[tid] = sharedRet[tid];
    // Multiply sharedA to sharedB and store in mutlDest
    mult(sharedA, sharedB, sharedRet);
    multDest[tid] = sharedRet[tid];
    
    // Divide sharedA by sharedB and store in divDest
    div(sharedA, sharedB, sharedRet);
    divDest[tid] = sharedRet[tid];
    
    // Mod sharedA by sharedB and store in modDest
    mod(sharedA, sharedB, sharedRet);
    modDest[tid] = sharedRet[tid];
}

// Executes all 5 global math operations
__global__ void executeGlobalMathOperations(int * a, int * b, int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{
    // Add a to b and store in addDest
    add(a, b, addDest);
    
    // Subtract a from b and store in subDest
    subtract(a, b, subDest);
    
    // Multiply a to b and store in mutlDest
    mult(a, b, multDest);
    
    // Divide a by b and store in divDest
    div(a, b, divDest);
    
    // Mod a by b and store in modDest
    mod(a, b, modDest);
}

// Executes all 5 register math operations
__global__ void executeRegisterMathOperations(int * a, int * b, int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{
    // Add a to b and store in addDest
    addReg(a, b, addDest);
    
    // Subtract a from b and store in subDest
    subtractReg(a, b, subDest);
    
    // Multiply a to b and store in mutlDest
    multReg(a, b, multDest);
    
    // Divide a by b and store in divDest
    divReg(a, b, divDest);
    
    // Mod a by b and store in modDest
    modReg(a, b, modDest);
}

// Executes all 5 constant math operations
__global__ void executeConstantMathOperations(int * addDest, int * subDest, int * multDest, int * divDest, int * modDest, const int size)
{    
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Add VAL_A to VAL_B and store in addDest
    addDest[tid] = VAL_A + VAL_B;
    
    // Subtract a from b and store in subDest
    subDest[tid] = VAL_A - VAL_B;
    
    // Multiply a to b and store in mutlDest
    multDest[tid] = VAL_A * VAL_B;
    
    // Divide a by b and store in divDest
    divDest[tid] = VAL_A / VAL_B; // B is chosen to not be 0.
    
    // Mod a by b and store in modDest
    modDest[tid] = VAL_A / VAL_B; // B is chosen to not be 0.
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

// Executes a streams test, which is similar to the GPU tests below except here we make use
// of CUDA streams and allocate/deallocate memory in an asynchronous fashion.
// The data is filled with random numbers that uses the same seed as the CPU tests.
void executeStreamTest(const int totalThreads, const int blockSize, const int numBlocks)
{    
    int a[totalThreads], b[totalThreads], add_dest[totalThreads], sub_dest[totalThreads], mult_dest[totalThreads], div_dest[totalThreads], mod_dest[totalThreads];

    int *gpu_a, *gpu_b, *gpu_add_dest, *gpu_sub_dest, *gpu_mult_dest, *gpu_div_dest, *gpu_mod_dest;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

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
    
    // Here we will now copy memory asynchronously and call each of the global version of the math
    // methods using a stream. This will allow the stream to do its own calculation of how these
    // methods should be executed.
    cudaMemcpyAsync(gpu_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice, stream);
 
    // Asynchronously add and then copy memory to host.
    addGlob<<<numBlocks, blockSize, 0, stream>>>(gpu_a, gpu_b, add_dest);
    cudaMemcpyAsync(add_dest,  gpu_add_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // Asynchronously subtract and then copy memory to host.
    subtractGlob<<<numBlocks, blockSize, 0, stream>>>(gpu_a, gpu_b, sub_dest);
    cudaMemcpyAsync(sub_dest,  gpu_sub_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // Asynchronously multiply and then copy memory to host.
    multGlob<<<numBlocks, blockSize, 0, stream>>>(gpu_a, gpu_b, mult_dest);
    cudaMemcpyAsync(mult_dest, gpu_mult_dest, totalThreads*sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // Asynchronously divide and then copy memory to host.
    divGlob<<<numBlocks, blockSize, 0, stream>>>(gpu_a, gpu_b, div_dest);
    cudaMemcpyAsync(div_dest,  gpu_div_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // Asynchronously modulous and then copy memory to host.
    modGlob<<<numBlocks, blockSize, 0, stream>>>(gpu_a, gpu_b, mod_dest);
    cudaMemcpyAsync(mod_dest,  gpu_mod_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_add_dest);
    cudaFree(gpu_sub_dest);
    cudaFree(gpu_mult_dest);
    cudaFree(gpu_div_dest);
    cudaFree(gpu_mod_dest); 
}

// Executes each of the gpu tests by creating local memory, copying it global memory, and then performing
// all 5 math operations on the data. 
// The data is filled with random numbers that uses the same seed as the CPU tests.
void executeGPUTest(const int totalThreads, const int blockSize, const int numBlocks, const gpu_tests_enum testType)
{
    // The stream test works differently enough that it requires a different method since its calls will all be async.
    if (testType == STREAM)
    {
        executeStreamTest(totalThreads, blockSize, numBlocks);
        return;
    }
    
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
    
    switch (testType)
    {
        case GLOBAL:
            // Executes global memory operations.
            executeGlobalMathOperations<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, numBlocks * blockSize);
            break;
        case SHARED:
            // The third parameter is the size of the shared memory
            // We multiply by 3 because we need to copy A and B and then also have room for the return in shared memory.
            executeSharedMathOperations<<<numBlocks, blockSize, 3 * totalThreads * sizeof(int)>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, totalThreads);
            break;
        case CONSTANT:
            // constant doesn't actually take in gpu_a and gpu_b since it uses constant memory. However the random generation is left in so timing can be compared.
            executeConstantMathOperations<<<numBlocks, blockSize>>>(gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, totalThreads);
            break;
        case REGISTER:
            // Executes global memory operations by saving the value into local registers first.
            executeRegisterMathOperations<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_add_dest,gpu_sub_dest,gpu_mult_dest,gpu_div_dest,gpu_mod_dest, totalThreads);
            break;
        default:
            std::cout << "Unknown test type " << testType << "!" << std::endl;
            break;
    }
    
    cudaMemcpy(add_dest,  gpu_add_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_dest,  gpu_sub_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(mult_dest, gpu_mult_dest, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(div_dest,  gpu_div_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mod_dest,  gpu_mod_dest,  totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_add_dest);
    cudaFree(gpu_sub_dest);
    cudaFree(gpu_mult_dest);
    cudaFree(gpu_div_dest);
    cudaFree(gpu_mod_dest); 
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
    
    for (auto testType = GLOBAL; testType < NUM_GPU_TESTS; ++testType)
    {
        startTime = std::chrono::system_clock::now();
        executeGPUTest(totalThreads, blockSize, numBlocks, testType);
        endTime = std::chrono::system_clock::now();
        totalTime = endTime-startTime;
        std::cout << gpu_tests_strings[testType] + " Memory execution took: " << totalTime.count() << " seconds." << std::endl;
    }
    
    return 0;
}
