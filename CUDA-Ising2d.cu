#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_free.h>
#include <thrust/transform.h>

const int J = 1.0;
const double K_b =1.0; //actual value : 1.3806503 x 10^(-23)
double T = 1.6;
const double finalTemp = 1.5;
const double tempStep = 0.1;

const int MCSTEPS = 1000000;
int numOfBins = 10;
int binSize = MCSTEPS/numOfBins;

const int num_elements_x = 128; //MUST BE MXM matrix, 2^n and no greater than 512*512
const int num_elements_y = 128;
const int MAXLENGTH = num_elements_x*num_elements_y;
const int BLOCKSIZE = num_elements_x/16;   ///CANNOT EXCEED 32*32 = 1024 theards  for 1024x1024 used num_elements_x/32




//nvcc  -arch=sm_20 -o Ising2d Ising2d.cu

using namespace std;

thrust::host_vector<float> energyPerStepBin;
thrust::host_vector<float> avgEnergyPerStepBin; 

thrust::host_vector<float> eneSquaredBin;
thrust::host_vector<float> avgEneSquaredBin;


thrust::host_vector<float> magnetizationPerStepBin; 
thrust::host_vector<float> avgMagnetizationPerStepBin;

thrust::host_vector<float> magSquaredBin;
thrust::host_vector<float> avgMagSquaredBin;

//MEASUREMENTS
vector<float> tempValues;
vector<float> energyMeasurements;
vector<float> eneSquaredMeasurements;
vector<float> magnetizationMeasurements;
vector<float> magSquaredMeasurements;

vector<float>magSuscept;
vector<float>specficHeat;


__device__ int calcStatesEnergy (float *array){
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



__global__ void getSitesEnegry(float *input)
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



float getStatesEnegry(thrust::device_vector<float> device_vector) {



  float* raw_ptr = thrust::raw_pointer_cast(device_vector.data());

  
  // create two dimensional 4x4 thread blocks
  dim3 block_size;
  block_size.x = BLOCKSIZE;
  block_size.y = BLOCKSIZE;

  // configure a two dimensional grid as well
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;

  cudaThreadSynchronize();


  // grid_size & block_size are passed as arguments to the triple chevrons as usual
  getSitesEnegry<<<grid_size,block_size>>>(raw_ptr);

  


  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(raw_ptr);
  float sum = thrust::reduce(dev_ptr, dev_ptr + MAXLENGTH);

  //thrust::device_free(dev_ptr);
  //cudaFree(raw_ptr);
  
  //return h_sum_array[0]/float(MAXLENGTH);

  return sum/MAXLENGTH;
}

float getStatesMag(int* host_array) {

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);

  //int num_bytes2 = num_elements_x * num_elements_y * sizeof(float);

  int *device_array;


  // allocate memory in either space
  cudaMalloc((void**)&device_array, num_bytes);
  // copy array to device
  cudaMemcpy(device_array ,host_array , num_bytes, cudaMemcpyHostToDevice);

  thrust::device_ptr<int> dev_ptr(device_array);
  float sum = thrust::reduce(dev_ptr, dev_ptr + MAXLENGTH);

    
  cudaFree(device_array);
  cudaDeviceReset();

  //return  h_sum_array[0]/float(MAXLENGTH);
  return sum/MAXLENGTH;
}



void printState(thrust::device_vector<float> host_array){
    double sum = 0;
  for(int row = 0; row < num_elements_y; ++row)
  {
    for(int col = 0; col < num_elements_x; ++col)
    {
      sum = sum + host_array[row*num_elements_x +col];
      int element = host_array[row * num_elements_x + col];
      if(host_array[row*num_elements_x +col] == -1 ){
          printf("\033[1;31m %2d \033[0m",  element);
        }
        else{
          printf("\033[1;36m %2d \033[0m", element);
        }

    }
    printf("\n");
  }
  printf("%f\n", sum/MAXLENGTH);
}


float getChangeEnergy(thrust::device_vector<float> device_vector, int i , int j){
  
  int m = num_elements_x - 1;
  int s = device_vector[i*num_elements_x + j];

  int neighbors, top, bottom, left, right; 
  
  //testing if on top/bottom edge. 
  if ( (i - 1) < 0 ){
    top = device_vector[m*num_elements_x + j]; 
  }else{
    top = device_vector[(i-1)*num_elements_x + j];
  };

  if ( (i + 1) > m){
    bottom = device_vector[ j ];
  }else{
    bottom = device_vector[(i+1)*num_elements_x + j];
  };

  //testing if on right/left edge. 
  if ( (j - 1) < 0 ){
    left = device_vector[i*num_elements_x + m]; 
  }else{
    left = device_vector[i*num_elements_x + (j-1)];
  };

  if ( (j + 1) > m){
    right = device_vector[i*num_elements_x ];
  }else{
    right = device_vector[i*num_elements_x + (j+1)];
  };

  neighbors = right+left+top+bottom;

  //printf("s %d : n %d : l %d : r %d : t %d : b %d\n", s, neighbors, left,right,top,bottom);

  float changeEne = 2*double(s)*double(neighbors)*J;
  
  return changeEne;
}


void warmUpSweep(thrust::device_vector<float> device_vector, int maxHeatingStep){

     for(int step = 0 ; step < maxHeatingStep ; step++ ){
     
      int i = rand()%num_elements_x;
      int j = rand()%num_elements_y;
      
      /*
      printf("i %d , j %d ", i ,j);
      int test = i*num_elements_y +j;
      printf(" %d \n",test);   
      */


      double delta_E = getChangeEnergy(device_vector, i , j);
      double boltzman = exp((-1.0*delta_E)/(K_b*T));

      //printf("delta E : %f  boltzman : %f\n", delta_E, boltzman);


      if(delta_E <= 0.0){

        device_vector[i*num_elements_x + j] = -1*device_vector[i*num_elements_x + j];  //flip spin
        //printf("the enegry at step %d is  : %f\n",maxHeatingStep,getStatesEnegry(device_vector));

      }else{

          double n =((double)rand()/(double)RAND_MAX);  //some voodoo to make random double between 0 to 1

          if(n <= boltzman){

            device_vector[i*num_elements_x + j] = -1*device_vector[i*num_elements_x + j];  //flip spin
            //printf("the enegry at step %d is  : %f\n",step,getStatesEnegry(device_vector));

        }
      } 
    //printf("the mag at step %d is  : %f\n",step,getStatesMag(device_vector));
    
  }

  
}


void monteCarloSweep(thrust::device_vector<float> device_vector, int MCSTEPS){

      clock_t start, end;

      start = clock();
      float E;
     for(int step = 0 ; step < MCSTEPS ; step++ ){
     
      int i = rand()%num_elements_x;
      int j = rand()%num_elements_y;
      
      /*
      printf("i %d , j %d ", i ,j);
      int test = i*num_elements_y +j;
      printf(" %d \n",test);   
      */

      
      double delta_E = getChangeEnergy(device_vector, i , j);
      double boltzman = exp((-1.0*delta_E)/(K_b*T));

      //printf("delta E : %f  boltzman : %f\n", delta_E, boltzman);


      if(delta_E <= 0.0){

        device_vector[i*num_elements_x + j] = -1*device_vector[i*num_elements_x + j];  //flip spin
        
        E = getStatesEnegry(device_vector);
        //energyPerStepBin.push_back(E);
        //printf("the enegry at step %d is  : %f\n",step,E);

      }else{

          double n =((double)rand()/(double)RAND_MAX);  //some voodoo to make random double between 0 to 1

          if(n <= boltzman){

            device_vector[i*num_elements_x + j] = -1*device_vector[i*num_elements_x + j];  //flip spin
            E = getStatesEnegry(device_vector);
            //energyPerStepBin.push_back(E);
            //printf("the enegry at step %d is  : %f\n",step,E);

        }else{

            E = getStatesEnegry(device_vector);
            //energyPerStepBin.push_back(E);
            //printf("the enegry at step %d is  : %f\n",step,E);

         }

      } 
    //printf("the mag at step %d is  : %f\n",step,getStatesMag(device_vector));
      /*
      while( energyPerStepBin.size() == binSize ){
        
        thrust::device_vector<float> D = energyPerStepBin;
        float E = thrust::reduce(D.begin(),D.end())/binSize;
        D.clear();
        D.shrink_to_fit();
        printf("the E is : %f\n",E);
        avgEnergyPerStepBin.push_back(E);

        //double E2 = average(eneSquaredBin);
        //avgEneSquaredBin.push_back(E2);

        //double M = average(magnetizationPerStepBin);
        //avgMagnetizationPerStepBin.push_back(M);

        //double M2 = average(magSquaredBin);
        //avgMagSquaredBin.push_back(M);

        energyPerStepBin.clear();
        energyPerStepBin.shrink_to_fit();
        //eneSquaredBin.clear();
        //magnetizationPerStepBin.clear();
        //magSquaredBin.clear();
    
      } 
    */
    
  }
  printf("the enegry  is  : %f\n",E);
  printState(device_vector);
  /*
  tempValues.push_back(T);

  thrust::device_vector<float> D = avgEnergyPerStepBin;
  float E = thrust::reduce(D.begin(),D.end())/avgEnergyPerStepBin.size();
  energyMeasurements.push_back(E);
  printf("the sweeps final E is : %f\n",E);
  D.clear();
  D.shrink_to_fit();

  end = clock();
  std::cout <<" +++++++++ MONTE CARLO FINISH at Temp : " << T << " +++++++++++++++ "<< std::endl;
  std::cout <<std::endl;
  std::cout<<" < E > : "<< energyMeasurements[energyMeasurements.size()-1] <<std::endl;
  */
  end = clock();

  std::cout<<" Total MCSweep Runtime: "<< double((end - start)) / double(CLOCKS_PER_SEC)<<"s"<<std::endl; 
  
}





int main(void){

  //int num_bytes = num_elements_x * num_elements_y * sizeof(float);
  //float *host_array = 0;

  thrust::host_vector<float> host_array(MAXLENGTH);
  // allocate memory in either space
  //host_array = (float*)malloc(num_bytes);

  for( int i = 0; i < MAXLENGTH; i++ ) {
    host_array[i] = (rand() % 2) * 2 - 1;
  }

 thrust::device_vector<float> device_vector = host_array;

  //printState(host_array);

  while(T> finalTemp){

  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("the temp is : %f\n",T);

  warmUpSweep(device_vector,MCSTEPS/100);
  monteCarloSweep(device_vector,MCSTEPS);
  //printState(host_array); 


    T = T - tempStep;

  }


}