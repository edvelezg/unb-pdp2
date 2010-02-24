/*******************************************************************
*
*    DESCRIPTION: This program performs the multiplication between
*    a matrix and a vector in a GPU
*
*    AUTHOR: Eduardo Gutarra Velez
*
*    DATE: 02/01/2010
*
*******************************************************************/

#include <iostream>
#include <ctime>
using namespace std;

/**
 * Number of threads per block
 */
const int blocksize = 16;
time_t seconds;

/*******************************************************************
*
*    Kernel Name: mult_matrix_by_vector
*
*    Parameters: inputs are: matrix a (size NxN ), array b (size N),
*    value of N. ouputs are: 
*
*    Description: This is the kernel to perform the multiplication
*    between a matrix and a vector. Each thread performs the dot
*    product between each row and the vector.
*
*******************************************************************/
__global__ void scan(float *g_odata, float *g_idata, int n)
{
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0, pin = 1;
// load input into shared memory.
// This is exclusive scan, so shift right by one and set first elt to 0
    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
    __syncthreads();
    for ( int offset = 1; offset < n; offset *= 2 )
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if ( thid >= offset )
            temp[pout*n+thid] += temp[pin*n+thid - offset];
        else
            temp[pout*n+thid] = temp[pin*n+thid];
        __syncthreads();
    }
    g_odata[thid] = temp[pout*n+thid]; // write output
}

__global__ void prescan(float *g_odata, float *g_idata, int n)
{
    extern __shared__ float temp[];// allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];
    for ( int d = n>>1; d > 0; d >>= 1 ) // build sum in place up the tree
    {
        __syncthreads();
        if ( thid < d )
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if ( thid == 0 )
    {
        temp[n - 1] = 0;
    } // clear the last element
    for ( int d = 1; d < n; d *= 2 ) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if ( thid < d )
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}

int main ( int argc, char *argv[] )
{

    /**
     * Command line arguments must be 1 which is the number of rows 
     * and columns for a matrix and the size of the vector. 
     */
//  if ( argc != 2 )
//  {
//      cout<<"usage: "<< argv[0] <<" <size n>\n";
//      return EXIT_FAILURE;
//  }

    int N = 16;
    float *a = new float[N];
    float *b = new float[N];
//  float *c = new float[N];

    seconds = time (NULL);
    srand(seconds);

    for ( int i = 0; i < N; ++i )
    {
        a[i] = (float) i*2.0;
        cout << "a[" << i << "]: " << a[i] << endl; 
    }

    for ( int i = 0; i < N; ++i )
    {
        // calculate a random number between 0 and 1000
        b[i] = (float) 10;
    }

    float *ad, *bd;
    const int sizeVec = N*sizeof(float);

    cudaMalloc( (void**)&ad, sizeVec );
    cudaMalloc( (void**)&bd, sizeVec );
    cudaMemcpy( ad, a, sizeVec, cudaMemcpyHostToDevice );

    dim3 dimBlock(blocksize);
    dim3 dimGrid(ceil(N/(float)blocksize));

    cout << endl;

    prescan<<<dimGrid, dimBlock>>>( bd, ad, N );

    cudaMemcpy( b, bd, sizeVec, cudaMemcpyDeviceToHost );

    /**
     * GPU Output.
     */

    for ( int i = 0; i < N; ++i )
    {
        cout << "b[" << i << "]: " << b[i] << endl; 
    }

    cudaFree( ad ); 
    cudaFree( bd ); 

    delete[] a;
    delete[] b;
    return 0;
}

