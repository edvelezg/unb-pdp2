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

using namespace std;

const int N = 1000;
const int blocksize = 16;

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

    if ( i < N )
    {
        for ( int e = 0; e < N; ++e )
        {
            index = e + i*N; //index = e + j*N;
            Cvalue += a[index]*b[e];  // res[i] = res[i] + matrix[i][j]*vec[j];
        }
        c[i] = Cvalue;
    }
}



int main() {

    float *a = new float[N*N];
    float *b = new float[N];
    float *c = new float[N];

    for ( int i = 0; i < N*N; ++i )
    {
        a[i] = 1.0f; 
    }

    for ( int i = 0; i < N; ++i )
    {
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
    dim3 dimGrid(N/blocksize);

    mult_matrix_by_vector<<<dimGrid, dimBlock>>>( ad, bd, cd, N );

    cudaMemcpy( c, cd, sizeVec, cudaMemcpyDeviceToHost );


//  for ( int i = 0; i < N; ++i )
//  {
//      for ( int j = 0; j < N; ++j )
//      {
//          int index = i + j*N;
//          cout << a[index] << " ";
//      }
//      cout << endl;
//  }

    for ( int i = 0; i < N; ++i ) {
        cout << "c[i]: " << c[i] << endl; 
    }


    cudaFree( ad ); 
    cudaFree( bd ); 
    cudaFree( cd );

    delete[] a; 
    delete[] b;
    delete[] c;
    return EXIT_SUCCESS;
}

