#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cuda.h>

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

  cudaMalloc((void**)&d_a, size);
  cudaMemcpy(d_a, h_a, size,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_b, size);
  cudaMemcpy(d_b, h_b, size,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_c, size);

	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // peform operation on GPU
  vecAddKernel<<<blockInfo->blocksPerGrid, blockInfo->threadsPerBlock>>>(d_a, d_b, d_c, len);
  // copy results back to CPU
  cudaMemcpy(h_c, d_c, size,  cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

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
    c[i] = a[i] + b[i];
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

int validateBlockInfoForDevice(CudaBlockInfo  * blockInfo, int deviceNum){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNum);

    if(prop.maxThreadsPerBlock < blockInfo->threadsPerBlock){
      printf("\nDevice %s is unable to process request!\n", prop.name);
      printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf("  Requested threads per block: %d\n", blockInfo->threadsPerBlock);
      return 0;
    }

    return 1;
}

void printUsage(){
  printf("Usage -- \n");
  printf("  VectorMultiply <lenght of vectors> <number of threads per block> <number of blocks per grid>\n");
}

int loadArguments(int argc, char * argv[], CudaBlockInfo * blockInfo, int * vecLength){

  if(argc != 4){
    printf("\nIncorrect number of arguments!\n\n");
  }
  else{
    if (isdigit(argv[1][0])){
      *vecLength = atoi(argv[1]);
      //printf("vec length %d\n", *vecLength);
    }
  
    if (isdigit(argv[2][0])){
      blockInfo->threadsPerBlock = atoi(argv[2]);
      //printf("threads per block %d\n", blockInfo->threadsPerBlock);
    }
  
    if (isdigit(argv[3][0])){
      blockInfo->blocksPerGrid = atoi(argv[3]);
      //printf("blocks per gird %d\n", blockInfo->blocksPerGrid);
    }

    // validate block and thread values from arguments
    if(!validateBlockInfoForDevice(blockInfo, 0)){
      return 0;
    }

    return 1;
  }

  printUsage();
  return 0;
}

int main(int argc, char * argv[])
{
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
  // process time variable
  float procTime;

  fillVecs(a, b, *vecLength);
//  printVecs(a, b, c, n);

  printf("\nVector multiplication using CPU with %d elements:\n", *vecLength);

  procTime = vecAddCPU(a, b, c, *vecLength);
  printVecs(a, b, c, *vecLength);
	printf("Time to calculate results on CPU: %f ms.\n", procTime);

  printf("\nVector multiplication using GPU witih %d elements, %d threads per block and %d blocks per grid:\n", 
         *vecLength, blockInfo->threadsPerBlock, blockInfo->blocksPerGrid);

  procTime = vecAddGPU(a, b, c, *vecLength, blockInfo);
  printVecs(a, b, c, *vecLength);
	printf("Time to calculate results on GPU: %f ms.\n", procTime);

  return 0;
}
