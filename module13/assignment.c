//
// Based on:
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define NUM_BUFFER_ELEMENTS 16

const std::unordered_set<std::string> validKernals = {
    "add",
    "sub",
    "mult",
    "div",
    "mod",
    "avg"    
};

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//    main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    auto startTime = std::chrono::system_clock::now();
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    int platform = DEFAULT_PLATFORM; 
    
    // Make all of these vector in order to handle any number of input commands
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    std::vector<float *> inputs0;    
    std::vector<float *> inputs1;   
    std::vector<float *> outputs;    
    std::vector<std::string> kernalNames;
    std::vector<cl_mem> buffers0;
    std::vector<cl_mem> buffers1;
    std::vector<cl_mem> outputBuffer;
    std::vector<cl_command_queue> queues;
    std::vector<cl_kernel> kernals;
    std::vector<cl_event> events;
    
    // Always start at 1 since argument 0 is the program name
    for (int i = 1; i < argc; ++i)
    {
        
        std::string kernalName(argv[i]);
        
        std::transform(kernalName.begin(), kernalName.end(), kernalName.begin(), ::tolower);
        
        if (validKernals.find(kernalName) == validKernals.end())
        {
            std::cout << kernalName << " is not a valid kernal name!" << std::endl;
        }
        else
        {
            kernalNames.push_back(kernalName);
        }
    }
    
    if (kernalNames.size() == 0)
    {
        std::cout << "No valid kernals were supplied. Exiting!" << std::endl;
        return 1;
    }
    
    // Need to resize these. We could try to push_back but that would cause issues
    // since everything is passed as pointers.
    contexts.resize(kernalNames.size());
    programs.resize(kernalNames.size());
    inputs0.resize(kernalNames.size());
    inputs1.resize(kernalNames.size());
    outputs.resize(kernalNames.size());
    buffers0.resize(kernalNames.size());
    buffers1.resize(kernalNames.size());
    outputBuffer.resize(kernalNames.size());
    queues.resize(kernalNames.size());
    kernals.resize(kernalNames.size());
    events.resize(kernalNames.size());
    
    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };
    
    for (size_t i = 0; i < kernalNames.size(); ++i)
    {
        const auto & kernalName = kernalNames[i];
        contexts[i] = clCreateContext(
            contextProperties, 
            numDevices,
            deviceIDs, 
            NULL,
            NULL, 
            &errNum);
        checkErr(errNum, "clCreateContext");
     
        // Create program from source
        programs[i] = clCreateProgramWithSource(
            contexts[i], 
            1,
            &src, 
            &length, 
            &errNum);
        checkErr(errNum, "clCreateProgramWithSource");
     
        // Build program
        errNum = clBuildProgram(
            programs[i],
            numDevices,
            deviceIDs,
            "-I.",
            NULL,
            NULL);
        
        // create input output buffers0
        inputs0[i] = new float[NUM_BUFFER_ELEMENTS * numDevices];
        for (unsigned int j = 0; j < NUM_BUFFER_ELEMENTS * numDevices; ++j)
        {
            inputs0[i][j] = j;
        }
        
        inputs1[i] = new float[NUM_BUFFER_ELEMENTS * numDevices];
        for (unsigned int j = 0; j < NUM_BUFFER_ELEMENTS * numDevices; ++j)
        {
            inputs1[i][j] = 2 * j + i;
        }
        
        outputs[i] = new float[NUM_BUFFER_ELEMENTS * numDevices];

        // create a single buffer to cover all the input data
        buffers0[i] = clCreateBuffer(
            contexts[i],
            CL_MEM_READ_WRITE,
            sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");
        
        // create a single buffer to cover all the input data
        buffers1[i] = clCreateBuffer(
            contexts[i],
            CL_MEM_READ_WRITE,
            sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");
        
        // create a single buffer to cover all the input data
        outputBuffer[i] = clCreateBuffer(
            contexts[i],
            CL_MEM_READ_WRITE,
            sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");

        // Create command queues
         queues[i] =  
            clCreateCommandQueue(
            contexts[i],
            deviceIDs[0],
            0,
            &errNum);
        checkErr(errNum, "clCreateCommandQueue");
     
        kernals[i] = clCreateKernel(
         programs[i],
         kernalName.c_str(),
         &errNum);
        checkErr(errNum, ("clCreateKernel(" + kernalName + ")").c_str());

        errNum = clSetKernelArg(kernals[i], 0, sizeof(cl_mem), (void *)&buffers0[i]);
        errNum = clSetKernelArg(kernals[i], 1, sizeof(cl_mem), (void *)&buffers1[i]);
        errNum = clSetKernelArg(kernals[i], 2, sizeof(cl_mem), (void *)&outputBuffer[i]);
        checkErr(errNum, ("clSetKernelArg(" + kernalName + ")").c_str());
     
        // Write input data
        errNum = clEnqueueWriteBuffer(
          queues[i],
          buffers0[i],
          CL_TRUE,
          0,
          sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
          (void*)inputs0[i],
          0,
          NULL,
          NULL);
          
        // Write input data
        errNum = clEnqueueWriteBuffer(
          queues[i],
          buffers1[i],
          CL_TRUE,
          0,
          sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
          (void*)inputs1[i],
          0,
          NULL,
          NULL);

        size_t gWI = NUM_BUFFER_ELEMENTS;

        errNum = clEnqueueNDRangeKernel(
          queues[i], 
          kernals[i], 
          1, 
          NULL,
          (const size_t*)&gWI, 
          (const size_t*)NULL, 
          0, 
          0, 
          &events[i]);
    }      
    
     //Wait for the last queue to complete before continuing on queue 0
     errNum = clEnqueueBarrier(queues[0]);
     errNum = clEnqueueWaitForEvents(queues[0], 1, &events.back());
    
    for (size_t i = 0; i < kernalNames.size(); ++i)
    {
        // Read back computed data
        clEnqueueReadBuffer(
                queues[i],
                outputBuffer[i],
                CL_TRUE,
                0,
                sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
                (void*)outputs[i],
                0,
                NULL,
                NULL);
     
        // Display output in rows
        for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
        {
         std::cout << " " << outputs[i][elems];
        }
        std::cout << std::endl;
    }
    
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "Execution took: " << totalTime.count() << " seconds." << std::endl;

    return 0;
}
