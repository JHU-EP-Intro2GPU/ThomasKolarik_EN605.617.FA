//Based on the work of Andrew Krepps
#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>

static const int CYPHER_OFFSET = 3;

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

__global__ void caesarCypher(char * textToEncrypt, const int offset)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    char c;
    c = textToEncrypt[thread_idx] - offset;
    
    // This assume the input array is all capital letters
    if (c < 'A')
    {
        c += 'Z' - 'A';
    }
    else if (c > 'Z')
    {
        c -= 'Z' - 'A';
    }
    textToEncrypt[thread_idx] = c;
}

void hostAdd(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

void hostSub(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

void hostMult(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

void hostMod(int * a, int * b, int *c, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] % b[i];
    }
}

void hostCaesarCypher(char * textToEncrypt, const int offset, const int size)
{
    
    for (int i = 0; i < size; ++i)
    {
        char c;
        c = textToEncrypt[i] - offset;
        
        // This assume the input array is all capital letters
        if (c < 'A')
        {
            c += 'Z' - 'A';
        }
        else if (c > 'Z')
        {
            c -= 'Z' - 'A';
        }
        textToEncrypt[i] = c;
    }
}

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
    
    // Mod all of the numbers c[i] = a[i] % b[i];
    hostMod(a,b,c, totalThreads);
    
    // Allocate space for a character array.
    int minChar = 'A';
    int maxChar = 'Z';
    std::uniform_int_distribution<int> charDist(minChar, maxChar);
    
    char textToEncrypt[totalThreads];
    
    for (int i = 0; i < totalThreads; ++i)
    {
        textToEncrypt[i] = charDist(gen);
    }
    
    hostCaesarCypher(textToEncrypt, totalThreads, CYPHER_OFFSET);
}

void executeGPUTest(const int totalThreads, const int blockSize, const int numBlocks)
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
    
    // Mod all of the numbers c[i] = a[i] % b[i];
    mod<<<numBlocks, blockSize>>>(gpu_a,gpu_b,gpu_c);
    
    cudaMemcpy(c, gpu_c, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    // Allocate space for a character array.
    int minChar = 'A';
    int maxChar = 'Z';
    std::uniform_int_distribution<int> charDist(minChar, maxChar);
    
    char textToEncrypt[totalThreads];
    
    for (int i = 0; i < totalThreads; ++i)
    {
        textToEncrypt[i] = charDist(gen);
    } 
    
    char * gpuTextToEncrypt;
    cudaMalloc((void**)&gpuTextToEncrypt, totalThreads * sizeof(char));
    cudaMemcpy(gpuTextToEncrypt, a, totalThreads * sizeof(char), cudaMemcpyHostToDevice);
    
    caesarCypher<<<numBlocks, blockSize>>>(gpuTextToEncrypt, CYPHER_OFFSET);
    
    cudaMemcpy(textToEncrypt, gpuTextToEncrypt, totalThreads*sizeof(int), cudaMemcpyDeviceToHost);
    
    
    cudaFree(textToEncrypt);
    
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
    std::cout << "Host execution took: " << totalTime.count() " seconds." << std::endl;
    
    startTime = std::chrono::system_clock::now();
    executeGPUTest(totalThreads, blockSize, numBlocks);
    endTime = std::chrono::system_clock::now();
    totalTime = endTime-startTime;
    std::cout << "GPU execution took: " << totalTime.count() " seconds." << std::endl;
    
    return 0;
}
