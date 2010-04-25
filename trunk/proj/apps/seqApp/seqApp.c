#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void runTest( unsigned int numElements );

int main( int argc, char** argv) 
{
//  double wallTime = wallClockTime();

    if ( argc != 2 )
    {
        printf("usage: %s <size n>\n", argv[0]);
        exit(-1);
    }

    int numElements = atoi(argv[1]); // number of elements 

    runTest( numElements );
    exit(EXIT_SUCCESS);
}

void runTest( unsigned int numElements )
{
    FILE *file;
    file = fopen("CPUtimes.txt","a+"); /* apend file (add text to */

    unsigned int memSize = sizeof( float) * numElements; // size of the memory
    unsigned int symMemSize = sizeof( char) * numElements; // size of the memory

	float* h_frequencies = (float*) malloc( memSize); // allocating input data
    char* h_symbols = (char*) malloc( symMemSize); // allocating input data

      // initalizing the memory with the elements
    for (unsigned int i = 0; i < numElements; ++i) 
    {
//  	unsigned int numUncompElems = h_exclusiveScan[0] + h_frequencies[numElements-1];
		h_frequencies[i] = (float) (i+1);
		// printf("i = %f\n", h_frequencies[i]);
    }
	
	// allocating symbolic data
    for (unsigned int i = 0; i < numElements; ++i) 
    {
		h_symbols[i] = 'A' + (char)(i%26); // (rand() & 0xf);
		// printf("i = %c\n", h_symbols[i]);
    }

    fclose(file); /*done!*/
}

