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
void mult_matrix_by_vector( float* a, float *b, float *c, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0;
    int index;

    /*
    * Each thread will perform the dot product between the row of the matrix 
    * and the vector that is being multiplied. 
    */
    if ( i < N )
    {
        for ( int e = 0; e < N; ++e )
        {
            index = e + i*N;
            Cvalue += a[index]*b[e];
        }
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

    /**
     * Command line arguments must be 1 which is the number of rows 
     * and columns for a matrix and the size of the vector. 
     */
    if ( argc != 2 )
    {
        cout<<"usage: "<< argv[0] <<" <size n>\n";
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    float *a = new float[N*N];
    float *b = new float[N];
    float *c = new float[N];

    seconds = time (NULL);
    srand(seconds);

    for ( int i = 0; i < N*N; ++i )
    {
        // calculate a random number between 0 and 1000
//      a[i] = (float) (rand()%RAND_MAX);
        a[i] = (float) i;
    }

    for ( int i = 0; i < N; ++i )
    {
        // calculate a random number between 0 and 1000
//      b[i] = (float) (rand()%RAND_MAX);
        b[i] = (float) i;
        c[i] = (float) 0;
    }

    float *ad, *bd, *cd;
    const int sizeVec = N*sizeof(float);
    const int sizeMat = N*sizeVec;

    cudaMalloc( (void**)&ad, sizeMat );
    cudaMalloc( (void**)&bd, sizeVec );
    cudaMalloc( (void**)&cd, sizeVec );
    cudaMemcpy( ad, a, sizeMat, cudaMemcpyHostToDevice );
    cudaMemcpy( bd, b, sizeVec, cudaMemcpyHostToDevice );

    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(N/(float)blocksize));

    /**
     * CPU-only version of the program.
     */
    //  int idx;
    //  for ( int i = 0; i < N; ++i )
    //  {
    //      for ( int j = 0; j < N; ++j )
    //      {
    //          idx = j + i*N;
    //          c[i] += a[idx]*b[j];
    //      }
    //  }
    //
    //  cout << "c[:" << 0 << "]=" << c[0] << endl;
    //  cout << "c[:" << 1 << "]=" << c[1] << endl;
    //  cout << "c[:" << 2 << "]=" << c[2] << endl;
    //  cout << "c[:" << 3 << "]=" << c[3] << endl;
    //  cout << "c[:" << 4 << "]=" << c[4] << endl;
    //
    //  cout << endl;

    mult_matrix_by_vector<<<dimGrid, dimBlock>>>( ad, bd, cd, N );

    cudaMemcpy( c, cd, sizeVec, cudaMemcpyDeviceToHost );

    /**
     * GPU Output.
     */
    cout << "c[" << 0 << "]=" << c[0] << endl;
    cout << "c[" << 1 << "]=" << c[1] << endl;
    cout << "c[" << 2 << "]=" << c[2] << endl;
    cout << "c[" << 3 << "]=" << c[3] << endl;
//  cout << "c[" << 4 << "]=" << c[4] << endl;

    cudaFree( ad ); 
    cudaFree( bd ); 
    cudaFree( cd );

    delete[] a;
    delete[] b;
    delete[] c;

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
    float* h_frequencies = (float*) malloc( memSize); // allocating input data
    char* h_symbols = (char*) malloc( memSize); // allocating input data

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
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
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
    
    free( h_frequencies);
    free( h_odata);
    // free( reference);
    CUDA_SAFE_CALL(cudaFree(d_frequencies));
    CUDA_SAFE_CALL(cudaFree(d_exclusiveScan));
}
