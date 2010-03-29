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
    float Cvalue = 0.0;

    /*
    * Each thread will perform the dot product between the row of the matrix 
    * and the vector that is being multiplied. 
    */
    if ( i < N )
    {
        c[i] = Cvalue;
    }
}

//  Each thread i writes a 1 to item X[i] in array A UNLESS X[i] == 0
//    (A is now [0 0 0 1 0 0 0 0 0 0 1 0 0])
__global__
void writeChangeLocations(float *x, float *c, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 1.0;
	int idx = x[i];

    /*
    * Each thread will perform the dot product between the row of the matrix 
    * and the vector that is being multiplied. 
    */
    if ( i < N && idx != 0)
    {
        c[idx] = Cvalue;
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

	// ======================================================================
	// = Stage 1: Exclusive-scan of F gives you X =
	// ======================================================================
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
    float* h_exclusiveScan = (float*) malloc( memSize);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_exclusiveScan, d_exclusiveScan, memSize,
                                cudaMemcpyDeviceToHost) );

	for(size_t i = 0; i < numElements; ++i)
	{
		printf("res: %f\n", h_exclusiveScan[i]);
	}
	
    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	// ======================================================================
	// = Stage 2: Loop over U threads Each thread i writes a 0 to item i in 
	// = array A (creates a list A of length U where all elements are zero)
	// ======================================================================

	unsigned int numUncompElems = h_exclusiveScan[numElements-1] + h_frequencies[numElements-1];
	unsigned int uncompMemSize = sizeof( float) * numUncompElems; // size of the memory
	
    // allocate device memory for exclusive scan output
    float* h_uncompressedArr = (float*) malloc( uncompMemSize);
    float* d_uncompressedArr; // final uncompressed proj idx
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompressedArr, uncompMemSize));

    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(numUncompElems/(float)blocksize));
		
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
	
	// ======================================================================
	// = Stage 3
	// ======================================================================
	
    dim3 dimGrid2(ceil(numElements/(float)blocksize)); // should be on the compressed array
    writeChangeLocations<<<dimGrid2, dimBlock>>>( d_exclusiveScan, d_uncompressedArr, numElements);

    CUDA_SAFE_CALL( cudaMemcpy( h_uncompressedArr, d_uncompressedArr, uncompMemSize, cudaMemcpyDeviceToHost));

	printf("Total Elements = %d\n", numUncompElems);
	printf("c[0]= %f\n", h_uncompressedArr[0]);
	printf("c[1]= %f\n", h_uncompressedArr[1]);
	printf("c[2]= %f\n", h_uncompressedArr[2]);
	printf("c[3]= %f\n", h_uncompressedArr[3]);
	printf("c[4]= %f\n", h_uncompressedArr[4]);
	printf("c[5]= %f\n", h_uncompressedArr[5]);
	printf("c[6]= %f\n", h_uncompressedArr[6]);
	printf("c[7]= %f\n", h_uncompressedArr[7]);
	printf("c[8]= %f\n", h_uncompressedArr[8]);
	printf("c[9]= %f\n", h_uncompressedArr[9]);
	printf("c[10]= %f\n", h_uncompressedArr[10]);
	printf("c[11]= %f\n", h_uncompressedArr[11]);
	printf("c[%d]= %f\n", numUncompElems-1, h_uncompressedArr[numUncompElems-1]);
	
	// ======================================================================
	// = Stage 4
	// ======================================================================
	
	CUDPPConfiguration config2;
    config2.op = CUDPP_ADD;
    config2.datatype = CUDPP_FLOAT;
    config2.algorithm = CUDPP_SCAN;
    config2.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan2 = 0;
    CUDPPResult result2 = cudppPlan(theCudpp, &scanplan2, config2, numUncompElems, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    cudppScan(scanplan2, d_uncompressedArr, d_uncompressedArr, numUncompElems);

    // allocate mem for the result on host side
    // float* h_exclusiveScan = (float*) malloc( memSize);
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_uncompressedArr, d_uncompressedArr, uncompMemSize,
                                cudaMemcpyDeviceToHost) );

	for(size_t i = 0; i < numUncompElems; ++i)
	{
		printf("res: %f\n", h_uncompressedArr[i]);
	}
	
    result2 = cudppDestroyPlan(scanplan2);
    if (CUDPP_SUCCESS != result2)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }
    

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_frequencies);
    free( h_exclusiveScan);
    // free( reference);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));
}
