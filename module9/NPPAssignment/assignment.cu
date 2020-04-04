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


void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

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
    pageRankTest(argc, argv);
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> totalTime = endTime-startTime;
    std::cout << "PageRank execution took: " << totalTime.count() << " seconds." << std::endl;
    
    return 0;
}
