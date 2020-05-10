#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <stdio.h>

#include <npp.h>

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>

// Counts the number of alive neighbors for a given square
// array: The array with the game world stored within it
// xSize: The size of the game world in the X direction
// ySize: The size of the game world in the Y direction
// xCoord: The X coordinate of the square to check
// yCoord: the Y coordinate of the square to check
// return: The number of squares around the square that are non-zero.
__device__ unsigned int countAliveNeighbors(const unsigned int * array, const unsigned int xSize, const unsigned int ySize, const unsigned int xCoord, const unsigned int yCoord)
{
    unsigned int aliveNeighbors = 0;
    
    // Since it is impossible to store an infinite game world we are going to 
    // have the game world wrap around that is, the coordinate (0,0) is directly
    // next to (xSize,0), and (0,ySize).
    const unsigned int x0 = (xCoord + xSize - 1) % xSize;
    const unsigned int x1 = (xCoord);
    const unsigned int x2 = (xCoord + 1) % xSize;
    
    const unsigned int y0 = (yCoord + ySize - 1) % ySize;
    const unsigned int y1 = (yCoord);
    const unsigned int y2 = (yCoord + 1) % ySize;
    
    // We unravel the obvious set of loops here to have to skip checking the
    // if condition of the neighbor equaling the current square.
    aliveNeighbors += (array[y0 * xSize + x0] != 0);
    aliveNeighbors += (array[y1 * xSize + x0] != 0);
    aliveNeighbors += (array[y2 * xSize + x0] != 0);
    aliveNeighbors += (array[y0 * xSize + x1] != 0);
    aliveNeighbors += (array[y2 * xSize + x1] != 0);
    aliveNeighbors += (array[y0 * xSize + x2] != 0);
    aliveNeighbors += (array[y1 * xSize + x2] != 0);
    aliveNeighbors += (array[y2 * xSize + x2] != 0);
    
    return aliveNeighbors;
}

// Completes a single iteration of the game world for a single cell
// array: The array with the game world stored within it
// xSize: The size of the game world in the X direction
// ySize: The size of the game world in the Y direction
// neighborsToGrow: The number of neighbors required for a cell to grow if previously dead.
// neighborsToDie: The number of neighbors at which the cell will die due to loneliness.
__global__ void progressTime(const unsigned int * array, unsigned int * result, const unsigned int xSize, const unsigned int ySize, const unsigned int neighborsToGrow, const unsigned int neighborsToDie)
{
    const unsigned int xCoord = ((blockIdx.x * blockDim.x) + threadIdx.x) % xSize;
    const unsigned int yCoord = ((blockIdx.x * blockDim.x) + threadIdx.x) / xSize;
    const unsigned int sqIndx = yCoord * xSize + xCoord;
    
    unsigned int aliveNeighbors = countAliveNeighbors(array, xSize, ySize, xCoord, yCoord);
    
    // This line wraps up all of the growing/dying mechanics of the game. In the normal game of life, neighrborsToGrow is 3
    // and neighborsToDie is 1. So the following reduces to aliveNeighbors == neighborsToGrow || (array[xCoord][yCoord] && aliveNeighbors == 2).
    result[sqIndx] = (aliveNeighbors == neighborsToGrow || (array[sqIndx] && aliveNeighbors > neighborsToDie)) && aliveNeighbors <= neighborsToGrow;
}

// Executes the device (gpu) version of the game of life algorithm.
// array: The array with the game world stored within it
// xSize: The size of the array in the X direction
// ySize: The size of the array in the Y direction
// neighborsToGrow: The number of neighbors required for a cell to grow if previously dead.
// neighborsToDie: The number of neighbors at which the cell will die due to loneliness.
void executeDevice(const unsigned int * array, const unsigned int xSize, const unsigned int ySize, const unsigned int neighborsToGrow, const unsigned int neighborsToDie)
{
    auto startTime = std::chrono::system_clock::now();
    unsigned int * result = (unsigned int*)calloc(xSize * ySize, sizeof(unsigned int));
    
    unsigned int * gpu_array;
    unsigned int * gpu_result;
    
    cudaMalloc((void**)&gpu_array,  xSize * ySize * sizeof(unsigned int));
    cudaMalloc((void**)&gpu_result, xSize * ySize * sizeof(unsigned int));
    
    cudaMemcpy(gpu_result, result, xSize * ySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    cudaFree(gpu_array);
    cudaFree(gpu_result);
    
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Device execution took: " << totalTime.count() << " seconds." << std::endl;
}

// Prints the given array out to the console
// array: The array with the game world stored within it
// xSize: The size of the array in the X direction
// ySize: The size of the array in the Y direction
// neighborsToGrow: The number of neighbors required for a cell
void printArray(const unsigned int * const array, const unsigned int xSize, const unsigned int ySize)
{
    for (size_t y = 0; y < ySize; ++y)
    {
        for(size_t x = 0; x < xSize; ++x)
        {
            std::cout << array[y * xSize + x] << " ";
        }
        std::cout << '\n';
    }
    
    std::cout << std::flush;
}

// Counts the number of alive neighbors for a given square
// array: The array with the game world stored within it
// xSize: The size of the game world in the X direction
// ySize: The size of the game world in the Y direction
// xCoord: The X coordinate of the square to check
// yCoord: the Y coordinate of the square to check
// return: The number of squares around the square that are non-zero.
unsigned int hostCountAliveNeighbors(const unsigned int * array, const unsigned int xSize, const unsigned int ySize, const unsigned int xCoord, const unsigned int yCoord)
{
    unsigned int aliveNeighbors = 0;
    
    // Since it is impossible to store an infinite game world we are going to 
    // have the game world wrap around that is, the coordinate (0,0) is directly
    // next to (xSize,0), and (0,ySize).
    const unsigned int x0 = (xCoord + xSize - 1) % xSize;
    const unsigned int x1 = (xCoord);
    const unsigned int x2 = (xCoord + 1) % xSize;
    
    const unsigned int y0 = (yCoord + ySize - 1) % ySize;
    const unsigned int y1 = (yCoord);
    const unsigned int y2 = (yCoord + 1) % ySize;
    
    // We unravel the obvious set of loops here to have to skip checking the
    // if condition of the neighbor equaling the current square.
    aliveNeighbors += (array[y0 * xSize + x0] != 0);
    aliveNeighbors += (array[y1 * xSize + x0] != 0);
    aliveNeighbors += (array[y2 * xSize + x0] != 0);
    aliveNeighbors += (array[y0 * xSize + x1] != 0);
    aliveNeighbors += (array[y2 * xSize + x1] != 0);
    aliveNeighbors += (array[y0 * xSize + x2] != 0);
    aliveNeighbors += (array[y1 * xSize + x2] != 0);
    aliveNeighbors += (array[y2 * xSize + x2] != 0);
    
    return aliveNeighbors;
}

// Completes a single iteration of the game world for a single cell
// array: The array with the game world stored within it
// xSize: The size of the game world in the X direction
// ySize: The size of the game world in the Y direction
// neighborsToGrow: The number of neighbors required for a cell to grow if previously dead.
// neighborsToDie: The number of neighbors at which the cell will die due to loneliness.
void hostProgressTime(const unsigned int * array, unsigned int * result, const unsigned int xSize, const unsigned int ySize, const unsigned int neighborsToGrow, const unsigned int neighborsToDie)
{
    for (unsigned int y = 0; y < ySize; ++y)
    {
        for (unsigned int x = 0; x < xSize; ++x)
        {
            const unsigned int sqIndx = y * xSize + x;
            unsigned int aliveNeighbors = hostCountAliveNeighbors(array, xSize, ySize, x, y);
            
            // This line wraps up all of the growing/dying mechanics of the game. In the normal game of life, neighrborsToGrow is 3
            // and neighborsToDie is 1. So the following reduces to aliveNeighbors == neighborsToGrow || (array[xCoord][yCoord] && aliveNeighbors == 2).
            result[sqIndx] = (aliveNeighbors == neighborsToGrow || (array[sqIndx] && aliveNeighbors > neighborsToDie)) && aliveNeighbors <= neighborsToGrow;
        }
    }
}

// Executes the host (cpu) version of the game of life algorithm.
// array: The array with the game world stored within it
// xSize: The size of the array in the X direction
// ySize: The size of the array in the Y direction
// neighborsToGrow: The number of neighbors required for a cell to grow if previously dead.
// neighborsToDie: The number of neighbors at which the cell will die due to loneliness.
void executeHost(const unsigned int * array, const unsigned int xSize, const unsigned int ySize, const unsigned int neighborsToGrow, const unsigned int neighborsToDie)
{
    auto startTime = std::chrono::system_clock::now();
    unsigned int * result = (unsigned int*)calloc(xSize * ySize, sizeof(unsigned int));
    hostProgressTime(array, result, xSize, ySize, neighborsToGrow, neighborsToDie);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Host execution took: " << totalTime.count() << " seconds." << std::endl;
    
    printArray(array, xSize, ySize);
    printArray(result, xSize, ySize);
}

// Takes in a file name and parses it to setup the initial game state.
// argc: The number of command line arguments. User should only ever input a single argument
// argv: Stores the command line arguments. The only user argument should be the file to read from.
int main(int argc, char** argv)
{
    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage("Example.pgn", oHostSrc);
    
    std::cout << oHostSrc.size().nHeight << std::endl;
    std::cout << oHostSrc.size().nWidth << std::endl;
    
    for (int i = 0; i < oHostSrc.size().nHeight; ++i)
    {
        for (int j = 0; j < oHostSrc.size().nWidth; ++j)
        {
            std::cout << oHostSrc.data()[i * oHostSrc.size().nWidth + j] << " ";
        }
        std::cout << std::endl;
    }
    /*// Argc should only have a single argument which is the name of the file to read.
    if (argc != 4)
    {
        std::cout << "Invalid number of arguments. Usage 'gameOfLife.exe # # Example.pgn' where the # are unsigned int for neighborsToGrow, and neighborsToDie respectively." << std::endl;
        
        return -1;
    }
    
    const unsigned int GROW_INDEX = 1;
    const unsigned int DIE_INDEX  = 2;
    const unsigned int PGN_INDEX  = 3;

    unsigned int xSize = 0;
    unsigned int ySize = 0;
    unsigned int neighborsToGrow = std::stoul(argc[GROW_INDEX]);
    unsigned int neighborsToDie = std::stoul(argc[DIE_INDEX]);

    unsigned int * array = (unsigned int*)calloc(xSize * ySize, sizeof(unsigned int));

    executeHost(array, xSize, ySize, neighborsToGrow, neighborsToDie);
    
    
    free(array);*/
    
    return 0;
}
