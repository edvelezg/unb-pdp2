#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <cutil_inline.h>

//nvcc.exe -I="C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK\C\common\inc" -L="C:\CUDA\lib" -lcudart -L="C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK\C\common\lib" -lcutil -ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin" -O2 -o %bd%rn %n%e

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////


void reduction_gold(float* odata, float* idata, const unsigned int len) 
{
  *odata = 0;
  for(int i=0; i<len; i++) *odata += idata[i];
}

void scan_gold(float* odata, float* idata, const unsigned int len) 
{
  odata[0] = 0;
  for(int i=1; i<len; i++) odata[i] = idata[i-1] + odata[i-1];
}


////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // Dynamically allocated shared memory for scan kernels

    extern  __shared__  float temp[];

    int thid = threadIdx.x;

    // Cache the computational window in shared memory

    temp[thid] = g_idata[thid] + g_idata[thid+blockDim.x];

    for (int d = blockDim.x>>1; d > warpSize; d >>= 1)
    {
      __syncthreads();
      if (thid<d)  temp[thid] += temp[thid+d];
    }

    __syncthreads();

    if (thid<warpSize) {
      for (int d = warpSize; d > 0; d >>= 1) {
        if (thid<d)       // this test can be commented out -- why?
          temp[thid] += temp[thid+d];
      }
    }

    if (thid==0) g_odata[0] = temp[0];
}


__global__ void scan(float *g_odata, float *g_idata)
{
    // Dynamically allocated shared memory for scan kernels

    extern  __shared__  float temp[];

    int ai, bi;
    int thid = threadIdx.x;

    // Cache the computational window in shared memory

    temp[1+thid]            = g_idata[thid];
    temp[1+thid+blockDim.x] = g_idata[thid+blockDim.x];

    __syncthreads();

    ai = 2*thid + 1;
    bi = ai + 1;

    temp[bi] += temp[ai];

    // build the sum in place up the tree

    for (int d = blockDim.x>>1; d > warpSize; d >>= 1)
    {
      __syncthreads();

      if (thid < d) {
        ai    = 2*ai;
        bi    = 2*bi;
        temp[bi] += temp[ai];
      }
    }

    __syncthreads();

    if (thid<warpSize) {
      for (int d = warpSize; d > 0; d >>= 1) {
        if (thid < d) {
          ai    = 2*ai;
          bi    = 2*bi;
          temp[bi] += temp[ai];
        }
      }
    }

    // scan back down the tree

    // clear the first element
    if (thid == 0) temp[0] = 0;

    bi = 2*ai - bi;

    // traverse down the tree building the scan in place

    __syncthreads();

    if (thid < warpSize) {
      for (int d = 1; d < 2*warpSize; d *= 2) {
        if (thid < d) {
          temp[ai] += temp[bi];
          ai = ai/2;
          bi = bi/2;
        }
      }
    }

    for (int d = 2*warpSize; d < blockDim.x; d *= 2)
    {
      __syncthreads();

      if (thid < d) {
        temp[ai] += temp[bi];
        ai = ai/2;
        bi = bi/2;
      }
    }

    __syncthreads();

    temp[ai] += temp[bi];

    __syncthreads();

    // write results to global memory
    g_odata[thid]            = temp[thid];
    g_odata[thid+blockDim.x] = temp[thid+blockDim.x];
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv) 
{
    int num_elements, num_threads, mem_size, shared_mem_size;

    float *h_data, *reference, sum;
    float *d_idata, *d_odata;

    cutilDeviceInit(argc, argv);

    num_elements = 512;
    num_threads  = num_elements / 2;
    mem_size     = sizeof(float) * num_elements;

    // allocate host memory to store the input data
    // and initialize to integer values between 0 and 1000

    h_data = (float*) malloc(mem_size);
      
    for(int i = 0; i < num_elements; i++) 
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

    // compute reference solutions

    reference = (float*) malloc(mem_size);
    scan_gold( reference, h_data, num_elements);
    reduction_gold(&sum, h_data, num_elements);

    // allocate device memory input and output arrays

    cutilSafeCall(cudaMalloc((void**)&d_idata, mem_size));
    cutilSafeCall(cudaMalloc((void**)&d_odata, mem_size));

    // copy host memory to device input array

    cutilSafeCall(cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

    // execute the kernel

    shared_mem_size = sizeof(float) * (num_elements+1);
    scan<<<1,num_threads,shared_mem_size>>>(d_odata,d_idata);
    cutilCheckMsg("scan kernel execution failed");

    // copy result from device to host

    cutilSafeCall(cudaMemcpy(h_data, d_odata, sizeof(float)*num_elements, 
                                                    cudaMemcpyDeviceToHost));

    // check results

    float err=0.0;
    for (int i = 0; i < num_elements; i++) {
      err += (h_data[i] - reference[i])*(h_data[i] - reference[i]);
    }
    printf("rms scan error  = %f\n",sqrt(err/num_elements));

    // execute the kernel

    shared_mem_size = sizeof(float) * num_elements;
    reduction<<<1,num_threads,shared_mem_size>>>(d_odata,d_idata);
    cutilCheckMsg("reduction kernel execution failed");

    // copy result from device to host

    cutilSafeCall(cudaMemcpy(h_data, d_odata, sizeof(float),
                                                    cudaMemcpyDeviceToHost));

    // check results

    printf("reduction error = %f\n",h_data[0]-sum);

    // cleanup memory

    free(h_data);
    free(reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
}


robot emits infrared
if (robot detects obstacle) {
  if (obstacle is to the left) {
    OS_Write(MotorProcessFifo, LEFT);
  }
  else if (obstacle is to the right) {
    OS_Write(MotorProcessFifo, RIGHT);
  }
}
else
{
  OS_Write(MotorProcessFifo, NONE);
}

OS_Read(MotorProcessFifo, &value)
if (value == LEFT) 
{
  move robot back
  send voltage to right wheel
}
else if (value == RIGHT) 
{
  move robot back
  send voltage to left wheel
}
else // Assumes the process has encountered no obstacles.
{
  send voltage to both wheels
}


while (true) 
{
  if (light is detected) 
  {
    OS_Wait(BuzzerSemaphore) //Waiting primitive of the operating system.
    robot beeps to the intensity of light.
  }
  else
  {
    OS_Signal(BuzzerSemaphore) //When no light is detected the process stops
                //occupying the piezo buzzer.
    OS_Yield()  
  }
}
	
	
Sporadic processes produce a pattern.
	Pattern is communicated from sporadic process and played 	by device process.
	Robot advances normally
	Robot  uses infrared  to detect if it has to make a turn.

Character = Produce Character()
OS_Wait(Message Queue Semaphore);
OS_Write(Buzzer Process Queue, character)
OS_Signal(MessageQueueSemaphore);

OS_Read(DeviceProcessQueue);

semaphore fillCount = 0
semaphore emptyCount = FIFOSIZE

sporadic process() {
    while (true) {
        character = produceCharacter()
        OS_Wait(emptyCount)
        OS_Write(character)
        OS_Signal(fillCount)
    }
 }

device buzzer process() {
    while (true) {
        OS_Wait(fillCount)
        OS_Read(character)
        OS_Signal(emptyCount)
        reproduce beep with character
    }
}

process A ()
{
  OS_Wait(LCD Semaphore);
  write message;
  OS_Signal(LCD Semaphore)
}
