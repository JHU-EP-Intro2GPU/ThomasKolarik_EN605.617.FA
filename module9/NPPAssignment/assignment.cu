/**
 * Most of the follow code falls under the following stipulation from NVIDIA.
 *
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <helper_string.h>
#include <helper_cuda.h>

// Checks the given nvGraphStatus for errors, then outputs
// a log and exits the program if any errors are produced.
void check_status(const nvgraphStatus_t & status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

// Prints the NPP library information
bool printfNPPinfo()
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

// Initializes a CUDA device and returns the device ID for future use.
int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

// Runs a box filter test using the given CUDA device parameters.
int boxFilterNPPTest(int argc, char **argv)
{
    try
    {
        std::string sFilename;
        char *filePath;

        cudaDeviceInit(argc, (const char **)argv);

        if (printfNPPinfo() == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_boxFilter.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create struct with box-filter mask size
        NppiSize oMaskSize = {5, 5};

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};

        // create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
        // set anchor point inside the mask to (oMaskSize.width / 2, oMaskSize.height / 2)
        // It should round down when odd
        NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

        // run box filter
                           nppiFilterBoxBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                                      oSrcSize, oSrcOffset,
                                                      oDeviceDst.data(), oDeviceDst.pitch(),
                                                      oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }
}

// Execute the NVIDIA example page rank test. This algorithm computes weights of various 
int pageRankTest(int argc, char **argv)
{
    const size_t  n = 6, nnz = 10, vertex_numsets = 3, edge_numsets = 1;
    const float alpha1 = 0.85, alpha2 = 0.90;
    const void *alpha1_p = (const void *) &alpha1, *alpha2_p = (const void *) &alpha2;
    int i, *destination_offsets_h, *source_indices_h;
    float *weights_h, *bookmark_h, *pr_1,*pr_2;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int cuda_device = 0;
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    if (deviceProp.major < 3)
    {
        printf("> nvGraph requires device SM 3.0+\n");
        printf("> Waiving.\n");
        exit(EXIT_WAIVED);
    }


    // Allocate host data
    destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
    source_indices_h = (int*) malloc(nnz*sizeof(int));
    weights_h = (float*)malloc(nnz*sizeof(float));
    bookmark_h = (float*)malloc(n*sizeof(float));
    pr_1 = (float*)malloc(n*sizeof(float));
    pr_2 = (float*)malloc(n*sizeof(float));
    vertex_dim = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    // Initialize host data
    vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1, vertex_dim[2]= (void*)pr_2;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]= CUDA_R_32F;
    
    weights_h [0] = 0.333333f;
    weights_h [1] = 0.500000f;
    weights_h [2] = 0.333333f;
    weights_h [3] = 0.500000f;
    weights_h [4] = 0.500000f;
    weights_h [5] = 1.000000f;
    weights_h [6] = 0.333333f;
    weights_h [7] = 0.500000f;
    weights_h [8] = 0.500000f;
    weights_h [9] = 0.500000f;

    destination_offsets_h [0] = 0;
    destination_offsets_h [1] = 1;
    destination_offsets_h [2] = 3;
    destination_offsets_h [3] = 4;
    destination_offsets_h [4] = 6;
    destination_offsets_h [5] = 8;
    destination_offsets_h [6] = 10;

    source_indices_h [0] = 2;
    source_indices_h [1] = 0;
    source_indices_h [2] = 2;
    source_indices_h [3] = 0;
    source_indices_h [4] = 4;
    source_indices_h [5] = 5;
    source_indices_h [6] = 2;
    source_indices_h [7] = 3;
    source_indices_h [8] = 3;
    source_indices_h [9] = 4;

    bookmark_h[0] = 0.0f;
    bookmark_h[1] = 1.0f;
    bookmark_h[2] = 0.0f;
    bookmark_h[3] = 0.0f;
    bookmark_h[4] = 0.0f;
    bookmark_h[5] = 0.0f;

    // Starting nvgraph
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    for (i = 0; i < 2; ++i)
        check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    // First run with default values
    check_status(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));
    
    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    printf("pr_1, alpha = 0.85\n"); for (i = 0; i<n; i++)  printf("%f\n",pr_1[i]); printf("\n");

    // Second run with different damping factor and an initial guess
    for (i = 0; i<n; i++)  
        pr_2[i] =pr_1[i]; 
   
    nvgraphSetVertexData(handle, graph, vertex_dim[2], 2);
    check_status(nvgraphPagerank(handle, graph, 0, alpha2_p, 0, 1, 2, 0.0f, 0));

    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[2], 2));
    printf("pr_2, alpha = 0.90\n"); for (i = 0; i<n; i++)  printf("%f\n",pr_2[i]); printf("\n");

    //Clean 
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(pr_2);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}

// Executes NVIDIA examples of using the NPP and nvgraph libraries
// These use the pagerank algorithm as well as the box filter algorithm.
int main(int argc, char** argv)
{    
    auto startTime = std::chrono::system_clock::now();
    pageRankTest(argc, argv);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "PageRank execution took: " << totalTime.count() << " seconds." << std::endl;
    
    
    startTime = std::chrono::system_clock::now();
    boxFilterNPPTest(argc, argv);
    endTime = std::chrono::system_clock::now();
    totalTime = endTime-startTime;
    std::cout << "boxFilter execution took: " << totalTime.count() << " seconds." << std::endl;
    
    return 0;
}
