/*******************************************************************
*
*    DESCRIPTION: This program performs the multiplication between
*    a matrix and a vector in a GPU
*
*    AUTHOR: Eduardo Gutarra Velez
*
*    DATE: 1/27/2010
*
*******************************************************************/

#include <iostream>
#include <ctime>
using namespace std;

/**
 * Number of threads that will run simultaneously in a block
 */
const int blocksize = 32;
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


int main ( int argc, char *argv[] )
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
        a[i] = (float) (rand()%RAND_MAX);
    }

    for ( int i = 0; i < N; ++i )
    {
        // calculate a random number between 0 and 1000
        b[i] = (float) (rand()%RAND_MAX);
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

    int idx;
    for ( int i = 0; i < N; ++i )
    {
        for ( int j = 0; j < N; ++j )
        {
            idx = j + i*N;
            c[i] += a[idx]*b[j];
//          cout << "c[:" << i << "]" << c[i] << endl;
        }
    }

    cout << "c[:" << 0 << "]=" << c[0] << endl; 
    cout << "c[:" << 1 << "]=" << c[1] << endl; 
    cout << "c[:" << 2 << "]=" << c[2] << endl; 
    cout << "c[:" << 3 << "]=" << c[3] << endl; 
    cout << "c[:" << 4 << "]=" << c[4] << endl; 

    cout << endl; 


    for ( int i = 0; i < N; ++i )
    {
        c[i] = (float) 0;
    }

    mult_matrix_by_vector<<<dimGrid, dimBlock>>>( ad, bd, cd, N );

    cudaMemcpy( c, cd, sizeVec, cudaMemcpyDeviceToHost );


    //  for ( int j = 0; j < N; ++j )
    //  {
    //      for ( int i = 0; i < N; ++i )
    //      {
    //          int index = i + j*N;
    //          cout << a[index] << " ";
    //      }
    //      cout << endl;
    //  }
    //

    cout << "c[" << 0 << "]=" << c[0] << endl; 
    cout << "c[" << 1 << "]=" << c[1] << endl; 
    cout << "c[" << 2 << "]=" << c[2] << endl; 
    cout << "c[" << 3 << "]=" << c[3] << endl; 
    cout << "c[" << 4 << "]=" << c[4] << endl; 

    //  for ( int j = 0; j < N; ++j ) {
    //      cout << "b[" << j << "]: " << b[j] << endl;
    //  }

    cudaFree( ad ); 
    cudaFree( bd ); 
    cudaFree( cd );

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
