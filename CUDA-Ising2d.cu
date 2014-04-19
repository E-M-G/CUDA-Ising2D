#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

const int J = 1.0;
const double K_b =1.0; //actual value : 1.3806503 x 10^(-23)
double T = 7.0;
const double finalTemp = 1.5;
const double tempStep = 0.1;

const int MCSTEPS = 100000000;

const int num_elements_x = 128; //MUST BE MXM matrix, 2^n and no greater than 512*512
const int num_elements_y = 128;
const int MAXLENGTH = num_elements_x*num_elements_y;
const int BLOCKSIZE = num_elements_x/4;

__device__ int stateArray[MAXLENGTH];


//nvcc  -arch=sm_20 -o Ising2d Ising2d.cu


__device__ int calcStatesEnergy (int *array){
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  //int index = index_y * grid_width + index_x;

  int stateEne = 0.0;
  int m = num_elements_x - 1;
  int s = array[index_y * grid_width + index_x];
  int neighbors, top, bottom, left, right; 

  if ((index_y -1) < 0){
    top = array[m * grid_width + index_x];
  }else{
    top = array[(index_y-1) * grid_width + index_x];
  }

  if ((index_y +1) > m) {
    bottom = array[index_x];
  }else{
    bottom = array[(index_y+1) * grid_width + index_x];
  }

  if ((index_x -1 )< 0){
    left = array[index_y * grid_width + m];
  }else{
    left = array[index_y * grid_width + (index_x - 1)];
  }

  if ((index_x + 1 ) > m){
    right = array[index_y * grid_width ];
  }else{
    right = array[index_y * grid_width + (index_x + 1)];
  }

  neighbors = right+left+top+bottom;
  stateEne = -s*neighbors*J;
  return stateEne;
}



__global__ void getSitesEnegry(int *input)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  // map the two 2D block indices to a single linear, 1D block index
  //int result = blockIdx.y * gridDim.x + blockIdx.x;

   input[index] = calcStatesEnergy(input) ;

}

/*
template<unsigned int blockSize>
__global__ void reduce(int *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    sdata[tid] = 0;


    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] +g_idata[i + blockSize];
        i+= gridSize;
    }
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        
        if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}
*/

float getStatesEnegry(int* host_array) {

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);

  int num_bytes2 = num_elements_x * num_elements_y * sizeof(float);

  int *device_array = 0;


  // allocate memory in either space
  cudaMalloc((void**)&device_array, num_bytes);

  float *d_sum_array;
  float *h_sum_array;

  h_sum_array = (float*)malloc(num_bytes2);
  cudaMalloc((void**)&d_sum_array, num_bytes2);

  
  // create two dimensional 4x4 thread blocks
  dim3 block_size;
  block_size.x = BLOCKSIZE;
  block_size.y = BLOCKSIZE;

  // configure a two dimensional grid as well
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;

  // copy array to device
  cudaMemcpy(device_array ,host_array , num_bytes, cudaMemcpyHostToDevice);

  // grid_size & block_size are passed as arguments to the triple chevrons as usual
  getSitesEnegry<<<grid_size,block_size>>>(device_array);


  cudaFree(device_array);

  // copy array to device
  cudaMemcpy(device_array ,host_array , num_bytes, cudaMemcpyHostToDevice);



  //reduce< BLOCKSIZE > <<< 1 ,block_size, num_bytes2 >>>(device_array, d_sum_array, MAXLENGTH);

  //cudaMemcpy(h_sum_array, d_sum_array, num_bytes2, cudaMemcpyDeviceToHost);

  //printf("the sum is : %f\n", h_sum_array[0]);

  

  //cudaFree(d_sum_array);   
 

  thrust::device_ptr<int> dev_ptr(device_array);
  float sum = thrust::reduce(dev_ptr, dev_ptr + MAXLENGTH);

  cudaFree(device_array);
  //return h_sum_array[0]/float(MAXLENGTH);

  return sum/MAXLENGTH;
}

float getStatesMag(int* host_array) {

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);

  int num_bytes2 = num_elements_x * num_elements_y * sizeof(float);

  int *device_array;


  // allocate memory in either space
  cudaMalloc((void**)&device_array, num_bytes2);

  float *d_sum_array;
  float *h_sum_array;

  h_sum_array = (float*)malloc(num_bytes2);
  cudaMalloc((void**)&d_sum_array, num_bytes2);

  // copy array to device
  cudaMemcpy(device_array ,host_array , num_bytes, cudaMemcpyHostToDevice);

  //dim3 block_size;
  //block_size.x = BLOCKSIZE;
 // block_size.y = BLOCKSIZE;

  //reduce< BLOCKSIZE > <<< 1 ,block_size, num_bytes2 >>>(device_array, d_sum_array, MAXLENGTH);

  //cudaMemcpy(h_sum_array, d_sum_array, num_bytes2, cudaMemcpyDeviceToHost);

  thrust::device_ptr<int> dev_ptr(device_array);
  float sum = thrust::reduce(dev_ptr, dev_ptr + MAXLENGTH);

  cudaFree(d_sum_array);   
  cudaFree(device_array);

  //return  h_sum_array[0]/float(MAXLENGTH);
  return sum/MAXLENGTH;
}



void printState(int* host_array){
    double sum = 0;
  for(int row = 0; row < num_elements_y; ++row)
  {
    for(int col = 0; col < num_elements_x; ++col)
    {
      sum = sum + host_array[row*num_elements_x +col];
      if(host_array[row*num_elements_x +col] == -1 ){
          printf("\033[1;31m %2d \033[0m", host_array[row * num_elements_x + col]);
        }
        else{
          printf("\033[1;36m %2d \033[0m", host_array[row * num_elements_x + col]);
        }

    }
    printf("\n");
  }
  printf("%f\n", sum/MAXLENGTH);
}


float getChangeEnergy(int* host_array, int i , int j){
  
  int m = num_elements_x - 1;
  int s = host_array[i*num_elements_x + j];

  int neighbors, top, bottom, left, right; 
  
  //testing if on top/bottom edge. 
  if ( (i - 1) < 0 ){
    top = host_array[m*num_elements_x + j]; 
  }else{
    top = host_array[(i-1)*num_elements_x + j];
  };

  if ( (i + 1) > m){
    bottom = host_array[ j ];
  }else{
    bottom = host_array[(i+1)*num_elements_x + j];
  };

  //testing if on right/left edge. 
  if ( (j - 1) < 0 ){
    left = host_array[i*num_elements_x + m]; 
  }else{
    left = host_array[i*num_elements_x + (j-1)];
  };

  if ( (j + 1) > m){
    right = host_array[i*num_elements_x ];
  }else{
    right = host_array[i*num_elements_x + (j+1)];
  };

  neighbors = right+left+top+bottom;

  //printf("s %d : n %d : l %d : r %d : t %d : b %d\n", s, neighbors, left,right,top,bottom);

  float changeEne = 2*double(s)*double(neighbors)*J;
  
  return changeEne;
}


void warmUpSweep(int *host_array, int maxHeatingStep){

     for(int step = 0 ; step < maxHeatingStep ; step++ ){
     
      int i = rand()%num_elements_x;
      int j = rand()%num_elements_y;
      
      /*
      printf("i %d , j %d ", i ,j);
      int test = i*num_elements_y +j;
      printf(" %d \n",test);   
      */


      double delta_E = getChangeEnergy(host_array, i , j);
      double boltzman = exp((-1.0*delta_E)/(K_b*T));

      //printf("delta E : %f  boltzman : %f\n", delta_E, boltzman);


      if(delta_E <= 0.0){

        host_array[i*num_elements_x + j] = -1*host_array[i*num_elements_x + j];  //flip spin
        //printf("the enegry at step %d is  : %f\n",maxHeatingStep,getStatesEnegry(host_array));

      }else{

          double n =((double)rand()/(double)RAND_MAX);  //some voodoo to make random double between 0 to 1

          if(n <= boltzman){

            host_array[i*num_elements_x + j] = -1*host_array[i*num_elements_x + j];  //flip spin
            //printf("the enegry at step %d is  : %f\n",step,getStatesEnegry(host_array));

        }
      } 
    //printf("the mag at step %d is  : %f\n",step,getStatesMag(host_array));
    
  }
  printState(host_array);
  printf("__________________________________________________________\n");
  printf("the enegry is  : %f\n",getStatesEnegry(host_array));
  printf("__________________________________________________________\n");
  printf("the mag is  : %f\n",getStatesMag(host_array));
  printf("\n");
  
}





int main(void){

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);
  int *host_array = 0;

  // allocate memory in either space
  host_array = (int*)malloc(num_bytes);

  for( int i = 0; i < MAXLENGTH; i++ ) {
     host_array[i] = (rand() % 2) * 2 - 1;
  }

 

  //printState(host_array);

  while(T> finalTemp){

  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("the temp is : %f\n",T);

  warmUpSweep(host_array,MCSTEPS);

  //printState(host_array); 


    T = T - tempStep;

  }

   printf("the final enegry at Temp %f is  : %f\n",T,getStatesEnegry(host_array));
   printf("the final mag is     : %f\n",getStatesMag(host_array));

}