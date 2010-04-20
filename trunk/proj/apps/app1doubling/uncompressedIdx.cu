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
// const int blocksize = 512;
time_t seconds;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
////////////////////////////////////////////////////////////////////////////////
void runTest( unsigned int multiplier );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{	
	if ( argc != 2 )
    {
		printf("usage: %s <size n>\n", argv[0]);
		exit(-1);
    }

    int multiplier = atoi(argv[1]); // number of elements 
		
	runTest( multiplier );
    
    // CUT_EXIT(argc, argv);
	exit(EXIT_SUCCESS);
}

void runTest( unsigned int multiplier )
{
	FILE *file;
	file = fopen("GPUtimes.txt","a+"); /* apend file (add text to */
	
	/* For timing purposes */
	cudaEvent_t start, stop;
	float elapsedTime[4];
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned int numElements = 1024;
    unsigned int memSize    = sizeof( float) * numElements; // size of the memory
    unsigned int symMemSize = sizeof( int) * numElements; // size of the memory
	
	// allocate host memory
    float* h_frequencies = (float*) malloc( memSize); // allocating input data
    int* h_symbols = (int*) malloc( symMemSize); // allocating input data    
	
    // allocate host memory
    int* h_uncompSymbols = (int*) malloc( memSize); // allocating input data

    // initalizing the memory with the elements
    for (int i = 0; i < numElements; ++i) 
    {
		h_frequencies[i] = (float) (std::pow(2.0, (int)multiplier));
		h_symbols[i] = 'A' + (int)(i%26); // (rand() & 0xf);
		// printf("i = %f\n", h_frequencies[i]);
    }

	unsigned int count = 0;
	
	for(int i = 0; i < numElements; ++i)
	{
		count += h_frequencies[i];
	}
	printf("count: %d\n", count);
	fprintf(file, "Number of elems: %d\n", count);
	fprintf(file, "memSize: %d\n", count*sizeof(int));
	
	
    symMemSize = sizeof( int) * count; // size of the memory
	
	// allocate host memory
    h_uncompSymbols = (int*) malloc( symMemSize ); // allocating input data

	int idx = 0;
    for (unsigned int i = 0; i < numElements; ++i) 
	{
		for(unsigned int j = 0; j < h_frequencies[i]; ++j)
		{
			h_uncompSymbols[idx] = h_symbols[i];
			idx++;
		}
	}

	// printf("Total Elements = %d\n", numUncomElems);	
    // allocate device memory for symbols
    int* d_uncompSymbols; // attribute values
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbols, symMemSize));
	cudaEventRecord( start, 0 );
	
    CUDA_SAFE_CALL( cudaMemcpy( d_uncompSymbols, h_uncompSymbols, symMemSize,
                                cudaMemcpyHostToDevice) );
	cudaEventRecord( stop, 0 );
	
	/* block until event actually recorded */
	cudaEventSynchronize( stop );	
	cudaEventElapsedTime( &elapsedTime[0], start, stop );
	
	CUDA_SAFE_CALL( cudaMemcpy( h_uncompSymbols, d_uncompSymbols, symMemSize,
                                cudaMemcpyDeviceToHost) );
    
	
	printf("h_uncompSymbols[0]= %d\n", h_uncompSymbols[0]);
    printf("h_uncompSymbols[1]= %d\n", h_uncompSymbols[1]);
    printf("h_uncompSymbols[2]= %d\n", h_uncompSymbols[2]);
    printf("h_uncompSymbols[3]= %d\n", h_uncompSymbols[3]);
    printf("h_uncompSymbols[4]= %d\n", h_uncompSymbols[4]);
    printf("h_uncompSymbols[5]= %d\n", h_uncompSymbols[5]);
    printf("h_uncompSymbols[6]= %d\n", h_uncompSymbols[6]);
    printf("h_uncompSymbols[7]= %d\n", h_uncompSymbols[7]);
    printf("h_uncompSymbols[8]= %d\n", h_uncompSymbols[8]);
    printf("h_uncompSymbols[9]= %d\n", h_uncompSymbols[9]);
    printf("h_uncompSymbols[%d]= %d\n", count-1, h_uncompSymbols[count-1]);
	fprintf(file, "Time to complete copying: %f\n", elapsedTime[0]);
    
	// // copy host memory to device
	//     CUDA_SAFE_CALL( cudaMemcpy( h_uncompSymbols, d_uncompSymbols, memSize,
	//                                 cudaMemcpyDeviceToHost) );
	
	CUDA_SAFE_CALL( cudaFree(d_uncompSymbols) );

    free(h_symbols);
    free(h_uncompSymbols);
    // fclose(file); /*done!*/
}
