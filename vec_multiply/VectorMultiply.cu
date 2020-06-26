#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cuda.h>

#define GPUNUM (0)

struct CudaBlockInfo{
  int threadsPerBlock;
  int blocksPerGrid;
};

__global__
void vecAddKernel(float * a, float * b, float * c, int size)
{
  int i = blockDim.x*blockIdx.x+threadIdx.x;

  if (i<size) c[i] = a[i] * b[i];

}

float vecAddGPU(float * h_a, float * h_b, float * h_c, int len, CudaBlockInfo * blockInfo){

	cudaEvent_t start, stop;
	float elapsedTime;

  int size = len * sizeof(float);

  float * d_a, * d_b, * d_c;

  // alocate memory and move vectors to GPU
  cudaMalloc((void**)&d_a, size);
  cudaMemcpy(d_a, h_a, size,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_b, size);
  cudaMemcpy(d_b, h_b, size,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_c, size);

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // peform operation on GPU
  vecAddKernel<<<blockInfo->blocksPerGrid, blockInfo->threadsPerBlock>>>(d_a, d_b, d_c, len);
  // copy results back to CPU
  cudaMemcpy(h_c, d_c, size,  cudaMemcpyDeviceToHost);

  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // free GPU memory
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  return elapsedTime;
}

float vecAddCPU(float * a, float * b, float * c, int size){
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  for(int i=0; i<size; ++i){
    c[i] = a[i] * b[i];
  }

	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  return elapsedTime;
}

void fillVecs(float * a, float * b, int size){
  for(int i=0; i<size; ++i){
    a[i] = i;
    b[i] = i;
  }
}

void printVecs(float * a, float * b, float * c, int len){
  for(int i=0; i<len; ++i){
    printf("a:%f b:%f, c:%f\n",a[i], b[i], c[i]);
  }

}

int cudaDeviceProperties(){
  // get number of cude devices
  int nDevices;
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if(err != cudaSuccess){
    printf("%s\n", cudaGetErrorString(err));
    return 0;
  }

  // if no cuda devices found, return error code
  if (nDevices < 1){
    return 0;
  }

  // print stats for each cuda device found
  for (int i = 0; i < nDevices; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwith (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  return 1;
}

int validateBlockInfoForDevice(CudaBlockInfo  * blockInfo, int vecLen, int deviceNum){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNum);

    if(prop.maxThreadsPerBlock < blockInfo->threadsPerBlock){
      printf("\nDevice %s is unable to process request!\n", prop.name);
      printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf("  Requested threads per block: %d\n", blockInfo->threadsPerBlock);
      return 0;
    }
    if((blockInfo->threadsPerBlock*blockInfo->blocksPerGrid) != vecLen){
      printf("Number of threads per block x Number of blocks per grid != Vector Length\n");
      return 0;
    }

    return 1;
}

void printUsage(){
  printf("\nUsage -- \n");
  printf("  VectorMultiply <length of vectors> <number of threads per block> <number of blocks per grid>\n");
}

bool isNumeric(char * str){
  int len = strlen(str);
  for(int i=0; i<len; ++i){
    if(!isdigit(str[i]))
        return false;
  }
  return true;
}

int loadArguments(int argc, char * argv[], CudaBlockInfo * blockInfo, int * vecLength){

  if(argc != 4){
    printf("\nIncorrect number of arguments!\n\n");
  }
  else if (!isNumeric(argv[1]) || !isNumeric(argv[2]) || !isNumeric(argv[3])){
    printf("\nNon-numeric value found in command line arguments\n\n");
  }
  else{
    // check if all arguments are integer values
    *vecLength = atoi(argv[1]);
    blockInfo->threadsPerBlock = atoi(argv[2]);
    blockInfo->blocksPerGrid = atoi(argv[3]);

    // validate block and thread values from arguments
    if(validateBlockInfoForDevice(blockInfo, *vecLength, GPUNUM)){
      return 1;
    }
  }

  printUsage();
  return 0;
}

int main(int argc, char * argv[])
{
  // variables for command line arguments
  int * vecLength = (int *)malloc(sizeof(int));
  CudaBlockInfo * blockInfo = (CudaBlockInfo *)malloc(sizeof(CudaBlockInfo));

  // check number of and types of command line arguments
  if(!loadArguments(argc, argv, blockInfo, vecLength)){
    return 1;
  }

  // identify cuda devices
  if(!cudaDeviceProperties()){
    return 1;
  }

  // vector variables
  float * a = (float*)malloc(*vecLength * sizeof(float));
  float * b = (float*)malloc(*vecLength * sizeof(float));
  float * c = (float*)malloc(*vecLength * sizeof(float));

  // fill vectors a and b with values
  fillVecs(a, b, *vecLength);

  printf("\nVector multiplication using CPU with %d elements:\n", *vecLength);

  float procTimeCPU = vecAddCPU(a, b, c, *vecLength);
  printVecs(a, b, c, *vecLength);

  printf("\nVector multiplication using GPU with %d elements, %d threads per block and %d blocks per grid:\n", 
         *vecLength, blockInfo->threadsPerBlock, blockInfo->blocksPerGrid);

  float procTimeGPU = vecAddGPU(a, b, c, *vecLength, blockInfo);
  printVecs(a, b, c, *vecLength);

  // print process times
	printf("\nTime to calculate results on CPU: %f ms.\n", procTimeCPU);
	printf("Time to calculate results on GPU: %f ms.\n", procTimeGPU);

  // free memory
  free(a); free(b); free(c);

  return 0;
}
