/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cutil.h"
#include "cudpp.h"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
// 
// extern "C" 
// void computeSumScanGold( float *reference, const float *idata, 
//                         const unsigned int len,
//                         const CUDPPConfiguration &config);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);

    unsigned int numElements = 32; // number of elements 
    unsigned int memSize = sizeof( float) * numElements; // size of the memory

    // allocate host memory
    float* h_idata = (float*) malloc( memSize); // allocating input data
    char* h_sdata = (char*) malloc( memSize); // allocating input data

    // initalizing the memory with the elements
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_idata[i] = (float) (i+1);
		printf("i = %f\n", h_idata[i]);
    }
	
	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_sdata[i] = 'A' + (char)i; // (rand() & 0xf);
		printf("i = %c\n", h_sdata[i]);
    }


    // allocate device memory
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, memSize,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    cudppScan(scanplan, d_odata, d_idata, numElements);

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, memSize,
                                cudaMemcpyDeviceToHost) );

	for(size_t i = 0; i < numElements; ++i)
	{
		printf("res: %f\n", h_odata[i]);
	}
	
    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_idata);
    free( h_odata);
    // free( reference);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
}
