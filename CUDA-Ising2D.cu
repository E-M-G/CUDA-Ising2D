#include <stdlib.h>
#include <stdio.h>

const int j = 1.0;
const int ROW = 1024;
const int COL = 1024;
const int MAXLENGTH = ROW*COL;
const int BLOCKSIZE = 256;


__device__ int calcStatesEnergy (int *array){
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  //int index = index_y * grid_width + index_x;

  int stateEne = 0.0;
  int m = ROW - 1;
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
  stateEne = -s*neighbors*j;
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
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;

  input[index] = calcStatesEnergy(input);
}






template<unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

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

int main(void){
    int num_bytes = MAXLENGTH * sizeof(int);



    int *device_array;
    int *d_sum_array;
    int *host_array;
    int *h_sum_array;
    // allocate memory in either space
    host_array = (int*)malloc(num_bytes);
    h_sum_array = (int*)malloc(num_bytes);
    cudaMalloc((void**)&device_array, num_bytes);
    cudaMalloc((void**)&d_sum_array, num_bytes);



    for( int i = 0; i < MAXLENGTH; i++ ) {
        host_array[i] = 1;
    }

    cudaMemcpy(device_array ,host_array , num_bytes, cudaMemcpyHostToDevice);


    reduce< BLOCKSIZE > <<< 1 ,BLOCKSIZE, BLOCKSIZE*sizeof(int) >>>(device_array, d_sum_array, MAXLENGTH);

    cudaMemcpy(h_sum_array, d_sum_array, num_bytes, cudaMemcpyDeviceToHost);

   /* for(int i = 0; i < num_elements_x; ++i){
       
        printf(" %d\n", host_array[i]);
    }*/

    printf("the sum is : %d\n", h_sum_array[0]);

    free(host_array);
    cudaFree(device_array);
     free(h_sum_array);
    cudaFree(d_sum_array);   
    
    return 0;
}