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
void uncompress(float *a, char *b, char *s, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int fromIdx = (int) a[i];
	int toIdx = (int) a[i+1];

    if ( i < N )
    {
		for(size_t j = fromIdx; j < toIdx; ++j)
		{
			b[j] = s[i];
		}
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
    unsigned int memSize = sizeof( float) * (numElements + 1); // size of the memory
    unsigned int symMemSize = sizeof( char) * numElements; // size of the memory

// allocate host memory
    float* h_frequencies = (float*) malloc( memSize); // allocating input data
    char* h_symbols = (char*) malloc( sizeof(char)*numElements); // allocating input data

// initalizing the memory with the elements
    for ( unsigned int i = 0; i < numElements; ++i )
    {
        h_frequencies[i] = (float) (i+1);
        // printf("i = %f\n", h_frequencies[i]);
    }
	h_frequencies[numElements] = 0; // Nice lil hack :)
	
// allocating symbolic data
    for ( unsigned int i = 0; i < numElements; ++i )
    {
        h_symbols[i] = 'A' + (char) i%26; // (rand() & 0xf);
        printf("i = %c\n", h_symbols[i]);
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

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(theCudpp, &scanplan, config, numElements+1, 1, 0);

    if ( CUDPP_SUCCESS != result )
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

// Run the scan
    cudppScan(scanplan, d_exclusiveScan, d_frequencies, numElements+1);

// allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
// copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_exclusiveScan, memSize,
                                cudaMemcpyDeviceToHost) );

	// for(size_t i = 0; i < numElements + 1; ++i)
	// {
	// 	printf("res: %f\n", h_odata[i]);
	// }

    result = cudppDestroyPlan(scanplan);

    if ( CUDPP_SUCCESS != result )
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	int numUncompElems = h_odata[numElements];
    unsigned int uncompMemSize = sizeof( char) * numUncompElems; // size of the memory

	char* h_uncompSymbArr = (char*) malloc (uncompMemSize);
	char* d_uncompSymbArr;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbArr, uncompMemSize));
	
    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(numUncompElems/(float)blocksize));

	uncompress<<<dimGrid, dimBlock>>>( d_exclusiveScan, d_uncompSymbArr, d_symbols, numElements);

    CUDA_SAFE_CALL( cudaMemcpy( h_uncompSymbArr, d_uncompSymbArr, uncompMemSize, cudaMemcpyDeviceToHost));

	/**
	* GPU Output.
	*/

    // printf("Total Elements = %d\n", numUncompElems);
    // printf("c[0]= %c\n", h_uncompSymbArr[0]);
    // printf("c[1]= %c\n", h_uncompSymbArr[1]);
    // printf("c[2]= %c\n", h_uncompSymbArr[2]);
    // printf("c[3]= %c\n", h_uncompSymbArr[3]);
    // printf("c[4]= %c\n", h_uncompSymbArr[4]);
    // printf("c[5]= %c\n", h_uncompSymbArr[5]);
    // printf("c[6]= %c\n", h_uncompSymbArr[6]);
    // printf("c[7]= %c\n", h_uncompSymbArr[7]);
    // printf("c[8]= %c\n", h_uncompSymbArr[8]);
    // printf("c[9]= %c\n", h_uncompSymbArr[9]);
    // printf("c[%d]= %c\n", numUncompElems-1, h_uncompSymbArr[numUncompElems-1]);

// shut down the CUDPP library
    cudppDestroy(theCudpp);

    free( h_frequencies);
    free( h_odata);
// free( reference);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));
}
