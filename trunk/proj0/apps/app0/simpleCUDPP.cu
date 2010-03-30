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

    if ( i < N && idx != 0)
    {
        c[idx] = Cvalue;
    }
}

__global__
void uncompress(float *a, char *b, char *s, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = (int) a[i];

    if ( i < N )
    {
        b[i] = s[idx];
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

    if ( argc != 2 )
    {
        cout<<"usage: "<< argv[0] <<" <size n>\n";
        return EXIT_FAILURE;
    }

    unsigned int numElements = atoi(argv[1]); // number of elements 
    unsigned int memSize = sizeof( float) * numElements; // size of the memory
    unsigned int symMemSize = sizeof( char) * numElements; // size of the memory

    // allocate host memory
    float* h_frequencies = (float*) malloc( memSize); // allocating input data
    char* h_symbols = (char*) malloc( symMemSize); // allocating input data

    // initalizing the memory with the elements
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_frequencies[i] = (float) (i+1);
		// printf("i = %f\n", h_frequencies[i]);
    }
	
	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_symbols[i] = 'A' + (char)i; // (rand() & 0xf);
		// printf("i = %c\n", h_symbols[i]);
    }

    // allocate device memory for frequencies
    float* d_frequencies; // frequencies
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_frequencies, memSize));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_frequencies, h_frequencies, memSize,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for symbols
    char* d_symbols; // attribute values
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_symbols, symMemSize));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_symbols, h_symbols, symMemSize,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for exclusive scan output
    float* d_exclusiveScan; // exclusive scan output
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_exclusiveScan, memSize));

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

	// ======================================================================
	// = Stage 1: Exclusive-scan of F
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
	
	// ======================================================================
	// = Stage 3
	// ======================================================================
	
    dim3 dimGrid2(ceil(numElements/(float)blocksize)); // should be on the compressed array
    writeChangeLocations<<<dimGrid2, dimBlock>>>( d_exclusiveScan, d_uncompressedArr, numElements);
	
	// ======================================================================
	// = Stage 4
	// ======================================================================
	
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
	result = cudppPlan(theCudpp, &scanplan, config, numUncompElems, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    cudppScan(scanplan, d_uncompressedArr, d_uncompressedArr, numUncompElems);
	
    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }
    
	// ======================================================================
	// = Stage 5
	// ======================================================================
	unsigned int uncompSymMemSize = sizeof( char) * numUncompElems; // size of the memory
	
    // allocate device memory for exclusive scan output
    char* h_uncompSymbArr = (char*) malloc( uncompSymMemSize);
    char* d_uncompSymbArr; // final uncompressed proj idx
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbArr, uncompSymMemSize));

    uncompress<<<dimGrid, dimBlock>>>( d_uncompressedArr, d_uncompSymbArr, d_symbols, numUncompElems);

	//     CUDA_SAFE_CALL( cudaMemcpy( h_uncompSymbArr, d_uncompSymbArr, uncompSymMemSize,
	//                                 cudaMemcpyDeviceToHost) );
	// 
	// for(size_t i = 0; i < numUncompElems; ++i)
	// {
	// 	printf("r[%d]: %c\n", i, h_uncompSymbArr[i]);
	// }
	
    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_frequencies);
    free( h_exclusiveScan);
    free( h_uncompSymbArr);
    free( h_uncompressedArr);
    free( h_symbols);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_uncompSymbArr));
    CUDA_SAFE_CALL(cudaFree(d_uncompressedArr));
    CUDA_SAFE_CALL(cudaFree(d_symbols));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));
}
