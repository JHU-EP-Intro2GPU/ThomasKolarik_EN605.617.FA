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

const int WHITE_PIXEL = 256;
const int BLACK_PIXEL = 0;
const int ALIVE = 1;
const int DEAD = 0;

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

// Reads a given PGM file and populates the given array, as well as the x and y sizes
// pgmName: The name of the PGM file to read.
// array: The array to populate with data.
// xSize: The xSize of the PGM.
// ySize: The ySize of the PGM.
void readPGM(const std::string & pgmName, unsigned int * &array, unsigned int & xSize, unsigned int & ySize)
{
    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(pgmName, oHostSrc);
    
    xSize = oHostSrc.size().nWidth;
    ySize = oHostSrc.size().nHeight;
    
    array = (unsigned int*)calloc(xSize * ySize, sizeof(unsigned int));
    
    // Put all of the image data into the given array.
    for (int y = 0; y < ySize; ++y)
    {
        for (int x = 0; x < xSize; ++x)
        {
            array[y * xSize + x] = oHostSrc.data()[y * xSize + x] == WHITE_PIXEL ? DEAD : ALIVE;
        }
    }
}

// Writes a PGM file with the given name.
// pgmName: The name of the PGM file to write.
// array: The array to pull data from.
// xSize: The xSize of the array.
// ySize: The ySize of the array.
void writePGM(const std::string & pgmName, const unsigned int * array, const unsigned int xSize, const unsigned int ySize)
{
    npp::ImageCPU_8u_C1 oHostDst(xSize, ySize);

    // Put all of the image data into the given array.
    for (int y = 0; y < ySize; ++y)
    {
        for (int x = 0; x < xSize; ++x)
        {
            // PGM white pixels (256) are dead and black pixels (0) are alive.
            oHostDst.data()[y * xSize + x] = array[y * xSize + x] != DEAD ? BLACK_PIXEL : WHITE_PIXEL;
        }
    }
    
    npp::saveImage(pgmName, oHostDst);
}

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
    aliveNeighbors += (array[y0 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y1 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y0 * xSize + x1] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x1] != DEAD);
    aliveNeighbors += (array[y0 * xSize + x2] != DEAD);
    aliveNeighbors += (array[y1 * xSize + x2] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x2] != DEAD);
    
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
    aliveNeighbors += (array[y0 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y1 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x0] != DEAD);
    aliveNeighbors += (array[y0 * xSize + x1] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x1] != DEAD);
    aliveNeighbors += (array[y0 * xSize + x2] != DEAD);
    aliveNeighbors += (array[y1 * xSize + x2] != DEAD);
    aliveNeighbors += (array[y2 * xSize + x2] != DEAD);
    
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
// iterations: The number of game iterations to run.
void executeHost(unsigned int * array, const unsigned int xSize, const unsigned int ySize, const unsigned int neighborsToGrow, const unsigned int neighborsToDie, const unsigned int iterations)
{
    auto startTime = std::chrono::system_clock::now();
    unsigned int ** results;
    results = (unsigned int**)calloc(iterations, sizeof(unsigned int *));
    
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        results[iter] = (unsigned int*)calloc(xSize * ySize, sizeof(unsigned int));
        hostProgressTime(array, results[iter], xSize, ySize, neighborsToGrow, neighborsToDie);
        memcpy(array, &results[iter][0], xSize * ySize * sizeof(unsigned int));
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Host execution took: " << totalTime.count() << " seconds." << std::endl;
    
    // Output the data as a PGM and then free up the memory.
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        writePGM("cpuIter" + std::to_string(iter) +".pgm", results[iter], xSize, ySize);
        free(results[iter]);
    }
    
    free(results);
}

// Takes in a file name and parses it to setup the initial game state.
// argc: The number of command line arguments.
// argv: Stores the command line arguments.
int main(int argc, char** argv)
{ 
    // Make sure we are inputing all of the correct arguments.
    if (argc != 5)
    {
        std::cout << "Invalid number of arguments. Usage 'gameOfLife.exe # # # Example.pgm' where the # are unsigned int for neighborsToGrow, neighborsToDie, and the number of iterations respectively." << std::endl;
        
        return -1;
    }
    
    const unsigned int GROW_INDEX = 1;
    const unsigned int DIE_INDEX  = 2;
    const unsigned int ITER_INDEX = 3;
    const unsigned int PGM_INDEX  = 4;

    unsigned int neighborsToGrow = std::stoul(argv[GROW_INDEX]);
    unsigned int neighborsToDie = std::stoul(argv[DIE_INDEX]);
    unsigned int iterations = std::stoul(argv[ITER_INDEX]);
    
    unsigned int * array = nullptr;
    unsigned int xSize = 0;
    unsigned int ySize = 0;
    
    readPGM(argv[PGM_INDEX], array, xSize, ySize);

    executeHost(array, xSize, ySize, neighborsToGrow, neighborsToDie, iterations);
    
    free(array);
    
    return 0;
}
