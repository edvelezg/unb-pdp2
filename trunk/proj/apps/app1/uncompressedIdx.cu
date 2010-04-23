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
void runTest( int numElements );

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

    int numElements = atoi(argv[1]); // number of elements 
		
	runTest( numElements );
    
    // CUT_EXIT(argc, argv);
	exit(EXIT_SUCCESS);
}

void runTest( int numElements ) 
{
	FILE *file;
	file = fopen("GPUtimes.txt","a+"); /* apend file (add text to */
	
	/* For timing purposes */
	cudaEvent_t start, stop;
	float elapsedTime[4];
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    unsigned int numUncomElems = (numElements*(numElements+1))/2; // number of elements 
    unsigned int memSize = sizeof( int) * numUncomElems; // size of the memory

	fprintf(file, "Number of elems: %d\n", numUncomElems);
	fprintf(file, "memSize: %d\n", numUncomElems);
	
    // allocate host memory
    int* h_symbols = (int*) malloc( numElements * sizeof(int)); // allocating input data
    int* h_uncompSymbols = (int*) malloc( memSize); // allocating input data
    int* h_frequencies = (int*) malloc( numElements * sizeof(int)); // allocating input data


	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_symbols[i] = 65 + (int)(i%26); // (rand() & 0xf);
		h_frequencies[i] = (float) (i+1);
		// printf("%d = %c\n", i, h_symbols[i]);
    }

	int idx = 0;
    for (unsigned int i = 0; i < numElements; ++i) 
	{
		for(unsigned int j = 0; j < h_frequencies[i]; ++j)
		{
			h_uncompSymbols[idx] = h_symbols[i];
			idx++;
		}
	}

	printf("Total Elements = %d\n", numUncomElems);
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
    printf("h_uncompSymbols[%d]= %d\n", numUncomElems-1, h_uncompSymbols[numUncomElems-1]);
	
    // allocate device memory for symbols
    int* d_uncompSymbols; // attribute values
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uncompSymbols, memSize));
	cudaEventRecord( start, 0 );
	
    CUDA_SAFE_CALL( cudaMemcpy( d_uncompSymbols, h_uncompSymbols, memSize,
                                cudaMemcpyHostToDevice) );
	cudaEventRecord( stop, 0 );
	
	/* block until event actually recorded */
	cudaEventSynchronize( stop );	
	cudaEventElapsedTime( &elapsedTime[0], start, stop );
	
	fprintf(file, "Time to complete copying: %f\n", elapsedTime[0]);
    
	// // copy host memory to device
	//     CUDA_SAFE_CALL( cudaMemcpy( h_uncompSymbols, d_uncompSymbols, memSize,
	//                                 cudaMemcpyDeviceToHost) );
	
	CUDA_SAFE_CALL( cudaFree(d_uncompSymbols) );

    free(h_symbols);
    free(h_uncompSymbols);
    // fclose(file); /*done!*/
}
