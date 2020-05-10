#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <stdio.h>

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
__global__ void progressTime(const unsigned int * array, unsinged int * result, const int xSize, const int ySize, const int neighborsToGrow, const int neighborsToDie)
{
    const unsigned int xCoord = ((blockIdx.x * blockDim.x) + threadIdx.x) % xSize;
    const unsigned int yCoord = ((blockIdx.x * blockDim.x) + threadIdx.x) / xSize;
    const int sqIndx = yCoord * xSize + xCoord;
    
    unisnged int aliveNeighbors = countAliveNeighbors(array, xSize, ySize, xCoord, yCoord);
    
    // This line wraps up all of the growing/dying mechanics of the game. In the normal game of life, neighrborsToGrow is 3
    // and neighborsToDie is 1. So the following reduces to aliveNeighbors == neighborsToGrow || (array[xCoord][yCoord] && aliveNeighbors == 2).
    result[sqIndx] = (aliveNeighbors == neighborsToGrow || (array[sqIndx] && aliveNeighbors > neighborsToDie)) && aliveNeighbors <= neighborsToGrow;
}

// Prints the given array out to the console
// array: The array with the game world stored within it
// xSize: The size of the array in the X direction
// ySize: The size of the array in the Y direction
// neighborsToGrow: The number of neighbors required for a cell
void printArray(const unsigned int * const array, const int xSize, const int ySize)
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

// Takes in a file name and parses it to setup the initial game state.
// argc: The number of command line arguments. User should only ever input a single argument
// argv: Stores the command line arguments. The only user argument should be the file to read from.
int main(int argc, char** argv)
{
    // Argc should only have a single argument which is the name of the file to read.
    if (argc != 2)
    {
        std::cout << "Invalid number of arguments. Usage 'gameOfLife.exe example.txt'." << std::endl;
        return -1;
    }
    
    ifstream fileToRead(argv[1]);
    
    if (!fileToRead.good())
    {
        std::cout << "Unable to read file " << argv[1] << std::endl;
        return -2;
    }
    
    sstream ss;
    
    ss << fileToRead.rdbuf();
    
    fileToRead.close();
    
    unsigned int xSize = 0;
    unsigned int ySize = 0;
    std::string str;
    while (ss >> str)
    {
        std::cout << str << std::endl;
    }
    
    /*ss >> xSize;
    ss >> ySize;
    
    unsigned int * array = (int*)calloc(xSize * ySize, sizeof(unsigned int));
    
    for (unsigned int y = 0; y < ySize; ++y)
    {
        for(unsigned int x = 0; x < xSize; ++x)
        {
            ss >> array[y * xSize + x];
        }
    }*/
    
    
    return 0;
}
