// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cutil.h"
#include "cudpp.h"

/**
* Number of threads per block
*/
const int blocksize = 512;
time_t seconds;

__global__
void uncompress(float *a, char *b, char *s, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int fromIdx = (int) a[i];
	int toIdx = (int) a[i+1];

    if ( i < N )
    {
		for(unsigned int j = fromIdx; j < toIdx; ++j)
		{
			b[j] = s[i];
		}
    }
}

double wallClockTime() { //time in seconds
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (1000000*tv.tv_sec+tv.tv_usec)/1.0e6;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( unsigned int numElements );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
	double wallTime = wallClockTime();
	
	if ( argc != 2 )
    {
		printf("usage: %s <size n>\n", argv[0]);
		exit(-1);
    }

    int numElements = atoi(argv[1]); // number of elements 
	
    
	FILE *file;
	
	file = fopen("CPUTimes.txt","a+"); /* apend file (add text to */
	// start = clock();
	
	runTest( numElements );
    
	wallTime = wallClockTime() - wallTime;
	
	fprintf(file,"%d time: %lf\n",numElements , wallTime); /*writes*/
    fclose(file); /*done!*/
    
    // CUT_EXIT(argc, argv);
	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( unsigned int numElements )
{
	FILE *file;
	file = fopen("GPUtimes.txt","a+"); /* apend file (add text to */
	
	/* For timing purposes */
	cudaEvent_t start, stop;
	float elapsedTime[7];
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
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
        h_symbols[i] = 'A' + (char) (i%26); // (rand() & 0xf);
        // printf("i = %c\n", h_symbols[i]);
    }

	// allocate device memory for frequencies
    float* d_frequencies; // frequencies
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_frequencies, memSize));
    // allocate device memory for symbols
    char* d_symbols; // attribute values
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_symbols, symMemSize));
	
	cudaEventRecord( start, 0 );
	// copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_frequencies, h_frequencies, memSize,
                                cudaMemcpyHostToDevice) );
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_symbols, h_symbols, symMemSize,
                                cudaMemcpyHostToDevice) );
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	/* block until event actually recorded */
	cudaEventElapsedTime( &elapsedTime[0], start, stop );
	fprintf(file, "Time to copy Compressed data: %f\n", elapsedTime[0]);

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

	cudaEventRecord( start, 0 );

// Run the scan
    cudppScan(scanplan, d_exclusiveScan, d_frequencies, numElements+1);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	/* block until event actually recorded */
	cudaEventElapsedTime( &elapsedTime[1], start, stop );
	fprintf(file, "Time to perform exclusive scan: %f\n", elapsedTime[1]);
	
	// allocate mem for the result on host side
	float* h_exclusiveScan = (float*) malloc( sizeof(float));
	// copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( &h_exclusiveScan[0], &d_exclusiveScan[numElements], sizeof(float),
                                cudaMemcpyDeviceToHost) );
	
    result = cudppDestroyPlan(scanplan);

    if ( CUDPP_SUCCESS != result )
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	int numUncompElems = h_exclusiveScan[0];
	
	fprintf(file, "total uncompressed elements: %d\n", numUncompElems);
	
    unsigned int uncompMemSize = sizeof( char) * numUncompElems; // size of the memory

	char* h_uncompSymbArr = (char*) malloc (uncompMemSize);
	char* d_uncompSymbArr;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbArr, uncompMemSize));
	
    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(numElements/(float)blocksize));

	cudaEventRecord( start, 0 );
	
	uncompress<<<dimGrid, dimBlock>>>( d_exclusiveScan, d_uncompSymbArr, d_symbols, numElements);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	/* block until event actually recorded */
	cudaEventElapsedTime( &elapsedTime[1], start, stop );
	fprintf(file, "Time to uncompress: %f\n", elapsedTime[1]);

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
    free( h_exclusiveScan);
// free( reference);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));

	/* Destroy the timer */
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	fclose(file); /*done!*/
}
