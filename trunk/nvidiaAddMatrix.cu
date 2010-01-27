#include <iostream>

using namespace std;

const int N = 16;
const int blocksize = 16;

__global__
void add_matrix_gpu( float* a, float *b, float *c, int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j*N;
    if ( i < N && j < N )
        c[index] = a[index] + b[index];
}



int main() {

    float *a = new float[N*N];
    float *b = new float[N*N];
    float *c = new float[N*N];

    for ( int i = 0; i < N*N; ++i ) {
        a[i] = 1.0f; b[i] = 3.5f;
    }

    float *ad, *bd, *cd;
    const int size = N*N*sizeof(float);

    cudaMalloc( (void**)&ad, size );
    cudaMalloc( (void**)&bd, size );
    cudaMalloc( (void**)&cd, size );
    cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );


    dim3 dimBlock( blocksize, blocksize );
    dim3 dimGrid(N/blocksize,N/blocksize);

    add_matrix_gpu<<<dimGrid, dimBlock>>>( ad, bd, cd, N );

    cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );

    cudaFree( ad ); cudaFree( bd ); cudaFree( cd );

    for ( int i = 0; i < N; ++i ) {
        for ( int j = 0; j < N; ++j ) {
            int index = i + j*N;
//          if ( index%1000==0 )
            cout << c[index] << " ";
        }
        cout << endl; 
    }

    delete[] a; 
    delete[] b;
    delete[] c;
    return EXIT_SUCCESS;
}

