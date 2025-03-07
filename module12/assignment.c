//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16
#define SUB_BUFFER_SIZE     2

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
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    auto progStartTime = std::chrono::system_clock::now();
    
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> subbuffers;
    
    int subbufferSize = SUB_BUFFER_SIZE;
    
    std::vector<float> inputValues;
    float * inputArray;
    float * outputArray;

    int platform = DEFAULT_PLATFORM; 

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;
    
    // Read in all of the input values.
    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--values"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            
            int numVals = 0;
            buffer >> numVals;
            
            // Skip to the first value
            ++i;
            int startI = i;
            
            // Read in the number of floating point values
            for (;i < startI + numVals && i < argc; ++i)
            {
                inputValues.push_back(std::stof(std::string(argv[i])));
            }
        }
        else
        {
            std::cout << "usage: --values n val1 val2 ... valn" << std::endl;
            return 0;
        }
    }


    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    // Do the initial program setup
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
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

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

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputArray = new float[NUM_BUFFER_ELEMENTS];
    outputArray = new float[NUM_BUFFER_ELEMENTS];
    
    // Populate the input arraty with either input values order
    // a simple counting pattern. If there weren't enough input elements
    // to fill the buffer then fill the rest with the counting pattern.
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
        if (i < inputValues.size())
        {
            inputArray[i] = inputValues[i];
        }
        else
        {
            inputArray[i] = i;
        }
    }

    // create a single buffer to cover all the input data
    cl_mem bufferInput = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        static_cast<void*>(inputArray),
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    
    // create a single buffer to cover all the output data
    cl_mem bufferOutput = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    
    // Create a number of subbuffers from the above buffer data based off the subbuffer size.
    int numSubbuffers = NUM_BUFFER_ELEMENTS / subbufferSize;

    // now for all devices other than the first create a sub-buffer
    for (int i = 0; i < numSubbuffers; i++)
    {
        cl_buffer_region region = 
            {
                subbufferSize * i * sizeof(float), 
                subbufferSize * sizeof(float)
            };
        cl_mem subbuffer = clCreateSubBuffer(
            bufferInput,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        subbuffers.push_back(subbuffer);
    }

    // Create a queue using all of the subbuffer data
    for (unsigned int i = 0; i < numSubbuffers; i++)
    {
        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[0],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "average",
            &errNum);
        checkErr(errNum, "clCreateKernel(average)");

        // Setup the arguments for the kernal
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&subbuffers[i]);
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferOutput);
        errNum = clSetKernelArg(kernel, 2, sizeof(cl_uint), &i);
        errNum = clSetKernelArg(kernel, 3, sizeof(cl_uint), &subbufferSize);
        checkErr(errNum, "clSetKernelArg(average)");

        kernels.push_back(kernel);
    }
        
    // Write input data
    errNum = clEnqueueWriteBuffer(
        queues[0],
        bufferInput,
        CL_TRUE,
        0,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        (void*)inputArray,
        0,
        NULL,
        NULL);

    std::vector<cl_event> events;
    // call kernel for each subbuffer
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = SUB_BUFFER_SIZE;

        errNum = clEnqueueNDRangeKernel(
            queues[i], 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

    // Read back computed data
    clEnqueueReadBuffer(
        queues[0],
        bufferOutput,
        CL_TRUE,
        0,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        (void*)outputArray,
        0,
        NULL,
        NULL);

    // Display output in rows
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
    {
        std::cout << " " << outputArray[elems];
    }

    std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;
    
    auto progEndTime = std::chrono::system_clock::now();
    std::chrono::duration<double> progTotalTime = progEndTime-progStartTime;
    std::cout << "Execution of entire program took: " << progTotalTime.count() << " seconds." << std::endl;

    return 0;
}
