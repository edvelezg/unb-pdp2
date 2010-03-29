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

/**
 * Number of threads per block
 */
const int blocksize = 32;
time_t seconds;

__global__
void initUncompressedArr(float *c, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 1234.0;

    /*
    * Each thread will perform the dot product between the row of the matrix 
    * and the vector that is being multiplied. 
    */
    if ( i < N )
    {
        c[i] = Cvalue;
    }
}

__global__
void writeChangeLocations(float *c, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0;

    /*
    * Each thread will perform the dot product between the row of the matrix 
    * and the vector that is being multiplied. 
    */
    if ( i < N )
    {
        c[i] = Cvalue;
    }
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
    // CUT_EXIT(argc, argv);
	exit(EXIT_SUCCESS);
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
    float* h_frequencies = (float*) malloc( memSize); // allocating input data
    char* h_symbols = (char*) malloc( sizeof(char)*numElements); // allocating input data

    // initalizing the memory with the elements
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_frequencies[i] = (float) (i+1);
		printf("i = %f\n", h_frequencies[i]);
    }
	
	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_symbols[i] = 'A' + (char)i; // (rand() & 0xf);
		printf("i = %c\n", h_symbols[i]);
    }

    // allocate device memory for frequencies
    float* d_frequencies; // frequencies
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_frequencies, memSize));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_frequencies, h_frequencies, memSize,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for exclusive scan output
    float* d_exclusiveScan; // exclusive scan output
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_exclusiveScan, memSize));

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    cudppScan(scanplan, d_exclusiveScan, d_frequencies, numElements);

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_exclusiveScan, memSize,
                                cudaMemcpyDeviceToHost) );

	// for(size_t i = 0; i < numElements; ++i)
	// {
	// 	printf("res: %f\n", h_odata[i]);
	// }
	
    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	unsigned int numUncompElems = h_odata[numElements-1];
	unsigned int uncompMemSize = sizeof( float) * numUncompElems; // size of the memory
	
    // allocate device memory for exclusive scan output
    float* h_uncompressedArr = (float*) malloc( uncompMemSize);
    float* d_uncompressedArr; // final uncompressed proj idx
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompressedArr, uncompMemSize));

    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(h_odata[numElements-1]/(float)blocksize));
	
	//CHANGED: Loop over U threads Each thread i writes a 0 to 
	// item i in array A (creates a list A of length U where all elements are zero)
    initUncompressedArr<<<dimGrid, dimBlock>>>( d_uncompressedArr, numUncompElems);

    CUDA_SAFE_CALL( cudaMemcpy( h_uncompressedArr, d_uncompressedArr, uncompMemSize, cudaMemcpyDeviceToHost));
	
	 /**
	  * GPU Output.
	  */
	
	printf("Total Elements = %d\n", numUncompElems);
	printf("c[0]= %f\n", h_uncompressedArr[0]);
	printf("c[1]= %f\n", h_uncompressedArr[1]);
	printf("c[2]= %f\n", h_uncompressedArr[2]);
	printf("c[%d]= %f\n", numUncompElems-1, h_uncompressedArr[numUncompElems-1]);

    // dim3 dimGrid2(ceil(h_odata[numElements-1]/(float)blocksize));

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_frequencies);
    free( h_odata);
    // free( reference);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));
}
