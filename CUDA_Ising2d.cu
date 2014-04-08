
const int j = 1.0;
const int ROW = 1024;
const int COL = 1024;
const int MAXLENGTH = ROW*COL;
const int BLOCKSIZE = 256;


extern void CUDA_Constructor(int** state_array, int size){
	
	cudaMalloc(state_array,sizeoof(int)*size);
}

extern void CUDA_setArray(int* d_state_array, h_state_array, int size){

	cudaMemcpy(d_state_array,h_state_array, sizeof(int)*size,cudaMemcopyHostToDevice);
}

extern void CUDA_getArray(int *h_state_array, int* d_state_array, int size){
	cudaMalloc(h_state_array, d_state_array, sizeof(int)*size , cudaMemcpyDevicetoHost);
}


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
  stateEne = -s*neighbors*j;
  return stateEne;
}



__global__ void getSitesEnegry(int *input, int* output)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  // map the two 2D block indices to a single linear, 1D block index
  int result = blockIdx.y * gridDim.x + blockIdx.x;

   output[index] = calcStatesEnergy(array) ;

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



extern int CUDA_getStatesEnegry(void){

	getSitesEnegry <<< BLOCKSIZE, GRIDSIZE >>> (d_state_array, d_ene_out_array);
	reduce< BLOCKSIZE > <<< 1 ,BLOCKSIZE, BLOCKSIZE*sizeof(int) >>>(device_array, d_sum_out_array, MAXLENGTH);

	return 0;

}
