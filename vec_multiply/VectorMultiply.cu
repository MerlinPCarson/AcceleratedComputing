#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cuda.h>

#define GPUNUM (0)

struct CudaBlockInfo{
  int threadsPerBlock;
  int blocksPerGrid;
};

// Function for checking for cuda errors
void cudaCheckError(cudaError_t err) {
  if(err!=cudaSuccess) {
    printf("Cuda failure %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
 }
}

__global__
void vecMultKernel(double * a, double * b, double * c, int size)
{
  int i = blockDim.x*blockIdx.x+threadIdx.x;

  if (i<size) c[i] = a[i] * b[i];

}

double vecMultGPU(double * h_a, double * h_b, double * h_c, int len, CudaBlockInfo * blockInfo){

	cudaEvent_t start, stop;
	float elapsedTime;

  int size = len * sizeof(double);
  double * d_a, * d_b, * d_c;
  double result = 0;

  // alocate memory and move vectors to GPU
  cudaCheckError(cudaMalloc((void**)&d_a, size));
  cudaCheckError(cudaMemcpy(d_a, h_a, size,  cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc((void**)&d_b, size));
  cudaCheckError(cudaMemcpy(d_b, h_b, size,  cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc((void**)&d_c, size));

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // peform operation on GPU
  vecMultKernel<<<blockInfo->blocksPerGrid, blockInfo->threadsPerBlock>>>(d_a, d_b, d_c, len);
  // copy results back to CPU
  cudaCheckError(cudaMemcpy(h_c, d_c, size,  cudaMemcpyDeviceToHost));

  for(int i=0; i<len; ++i){
    result += h_c[i];
  }

  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // print process time
	printf("Time to calculate results on GPU: %f ms.\n", elapsedTime);

  // free GPU memory
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  return result;
}

double vecMultCPU(double * a, double * b, double * c, int len){
	cudaEvent_t start, stop;
	float elapsedTime;
  double result = 0;

	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // do vector multiplication
  for(int i=0; i<len; ++i){
    c[i] = a[i] * b[i];
    result += c[i];
  }

	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // print process time
	printf("Time to calculate results on CPU: %f ms.\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  return result;
}

void fillVecs(double * a, double * b, int size){
  for(int i=0; i<size; ++i){
    a[i] = i;
    b[i] = i;
  }
}

void printVecs(double * a, double * b, double * c, int len){
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

  double bytesInGiB = 1 << 30;

  // print stats for each cuda device found
  for (int i = 0; i < nDevices; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory (GiB): %lf\n", prop.totalGlobalMem/bytesInGiB);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Maximum x Dimension of Grid: %d\n", prop.maxGridSize[0]);
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

    if((blockInfo->threadsPerBlock*blockInfo->blocksPerGrid) < vecLen){
      printf("Number of threads per block x Number of blocks per grid < Vector Length\n");
    }
    else if(prop.maxThreadsPerBlock < blockInfo->threadsPerBlock){
      printf("\nDevice %s is unable to process request!\n", prop.name);
      printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf("  Requested threads per block: %d\n", blockInfo->threadsPerBlock);
    }
    else if(prop.maxGridSize[0] < blockInfo->blocksPerGrid){
      printf("\nDevice %s is unable to process request!\n", prop.name);
      printf("  Max blocks per grid: %d\n", prop.maxGridSize[0]);
      printf("  Requested blocks per grid: %d\n", blockInfo->blocksPerGrid);
    }
    else if(prop.totalGlobalMem < 3*(vecLen*sizeof(double))){
      printf("\nDevice %s is unable to process request!\n", prop.name);
      printf("  Total global memory is: %lu\n", prop.totalGlobalMem);
      printf("  Bytes needed for vectors: %lu\n", 3*(vecLen*sizeof(double)));
    }
    else{
      return 1;
    }

    // validation failed
    return 0;
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
  double * a = (double*)malloc(*vecLength * sizeof(double));
  double * b = (double*)malloc(*vecLength * sizeof(double));
  double * c = (double*)malloc(*vecLength * sizeof(double));
  if(a == NULL || b == NULL || c == NULL){
    perror("malloc");
    exit(EXIT_FAILURE);
  }

  double result = 0;

  // fill vectors a and b with values
  fillVecs(a, b, *vecLength);

  printf("\nVector multiplication using CPU with %d elements:\n", *vecLength);

  result = vecMultCPU(a, b, c, *vecLength);
  printf("Result of vector multiplication on the CPU: %.2lf\n", result);

  printf("\nVector multiplication using GPU with %d elements, %d threads per block and %d blocks per grid:\n", 
         *vecLength, blockInfo->threadsPerBlock, blockInfo->blocksPerGrid);

  result = vecMultGPU(a, b, c, *vecLength, blockInfo);
  printf("Result of vector multiplication on the GPU: %.2lf\n", result);

  // free memory
  free(a); free(b); free(c);
  free(vecLength); free(blockInfo);

  return 0;
}
