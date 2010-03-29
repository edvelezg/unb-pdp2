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
// const int blocksize = 32;
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

    unsigned int numElements = 32; // number of elements 
    unsigned int numUncomElems = (numElements*(numElements+1))/2; // number of elements 
    unsigned int memSize = sizeof( char) * numUncomElems; // size of the memory

    // allocate host memory
    char* h_symbols = (char*) malloc( memSize); // allocating input data
    char* h_uncompSymbols = (char*) malloc( numUncomElems); // allocating input data


	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_symbols[i] = 'A' + (char)i; // (rand() & 0xf);
		// printf("i = %c\n", h_symbols[i]);
    }

    // initalizing the memory with the elements
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		for(size_t j = 0; j < i+1; ++j)
		{
			printf("sym[%d] = %c\n", j, h_symbols[i]);
			
		}
    }
	
    // allocate device memory for symbols
    char* d_uncompSymbols; // attribute values
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbols, numUncomElems));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_uncompSymbols, h_uncompSymbols, memSize,
                                cudaMemcpyHostToDevice) );

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);
		
    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_symbols);
    CUDA_SAFE_CALL(cudaFree(d_symbols));
}
