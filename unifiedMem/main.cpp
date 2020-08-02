#include <stdlib.h>
#include <random>
#include <cuda.h>


#define BLOCK_WIDTH (32)

// Macro for checking for cuda errors
#define cudaCheckError(status) 									\
do {												\
  cudaError_t err = status;									\
  if(err!=cudaSuccess) {									\
    printf("Cuda failure %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
    exit(EXIT_FAILURE);										\
  }												\
 } while(0)					

float matmul(float * M, float * N, float * P, int Mrows, int Mcols, int Nrows, int Ncols ){

	cudaEvent_t start, stop;
	float elapsedTime;

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // iterate over output dimensions
  for(int i = 0; i < Mrows; ++i){
    for(int j = 0; j < Ncols; ++j){
      float val = 0.0;
      for(int k = 0; k < Mcols; ++k){
        val += M[i*Mcols+k] * N[k*Ncols+j];
      }
      P[i*Ncols+j] = val;
    }
  }
  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  return elapsedTime;
}

__global__
void matmulKernel(float * M, float * N, float * P, int Mrows, int Mcols, int Nrows, int Ncols){

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < Mrows && col < Ncols){
    float val = 0.0;
    for(int i = 0; i < Mcols; ++i){
      val += M[row*Mcols+i] * N[i*Ncols+col];
    }
    P[row*Ncols+col] = val;
  }

}

float matmulDevice(float * M, float * N, float * P, int Mrows, int Mcols, int Nrows, int Ncols, int unified){

	cudaEvent_t start, stop;
	float elapsedTime;

  float * d_M, * d_N, * d_P;

  if(unified == 0){
    // alocate memory and move vectors to GPU
    cudaCheckError(cudaMalloc((void**)&d_M, Mrows*Mcols*sizeof(float)));
    cudaCheckError(cudaMemcpy(d_M, M, Mrows*Mcols*sizeof(float),  cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc((void**)&d_N, Nrows*Ncols*sizeof(float)));
    cudaCheckError(cudaMemcpy(d_N, N, Nrows*Ncols*sizeof(float),  cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc((void**)&d_P, Mrows*Ncols*sizeof(float)));
  }

  dim3 gridDim((int)ceil((float)Ncols / BLOCK_WIDTH), (int)ceil((float)Mrows / BLOCK_WIDTH), 1);
  dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  if(unified == 0){
    matmulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, Mrows, Mcols, Nrows, Ncols);
    cudaCheckError(cudaMemcpy(P, d_P, Mrows*Ncols*sizeof(float),  cudaMemcpyDeviceToHost));
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_N);
  } else {
    matmulKernel<<<gridDim, blockDim>>>(M, N, P, Mrows, Mcols, Nrows, Ncols);
  }

  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  return elapsedTime;
}

void printMatrix(float * matrix, int rows, int cols){

  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      printf("%f ", matrix[i*cols+j]);
    }
    printf("\n");
  }

}

void fillMatrix(float * matrix, int rows, int cols){
  int maxVal = 4;

  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      matrix[i*cols+j] = rand() % maxVal;
    }
  }

}

int main(void){

  // use unified memory for GPU kernel?
  int unified = 0;

  float elapsedTime = 0.0;

  int Mrows = 9;
  int Mcols = 7;
  int Nrows = Mcols;
  int Ncols = 11;

  float * M = (float*)malloc(Mrows*Mcols*sizeof(float));
  float * N = (float*)malloc(Nrows*Ncols*sizeof(float));
  float * P = (float*)calloc(Mrows*Ncols, sizeof(float));

  fillMatrix(M, Mrows, Mcols);
  printf("Matrix M: \n");
  printMatrix(M, Mrows, Mcols);

  fillMatrix(N, Nrows, Ncols);
  printf("Matrix N: \n");
  printMatrix(N, Nrows, Ncols);
 
// Problem 1a 
  printf("\nProblem 1a: sequential matrix multiplication\n");
  elapsedTime = matmul(M, N, P, Mrows, Mcols, Nrows, Ncols);
  printf("Matrix M*N: \n");
  printMatrix(P, Mrows, Ncols);
	printf("Time to calculate results on CPU: %f ms.\n", elapsedTime);


// Problem 1b 
  printf("\nProblem 1b: matrix multiplication using CUDA\n");
  elapsedTime = matmulDevice(M, N, P, Mrows, Mcols, Nrows, Ncols, unified);
  printf("Matrix M*N: \n");
  printMatrix(P, Mrows, Ncols);
	printf("Time to calculate results on GPU: %f ms.\n", elapsedTime);


// Problem 1c 
  printf("\nProblem 1c: matrix multiplication using unified memory\n");
  // enable use of unified memory
  unified = 1;

  cudaCheckError(cudaMallocManaged(&M, Mrows*Mcols*sizeof(float)));
  cudaCheckError(cudaMallocManaged(&N, Nrows*Ncols*sizeof(float)));
  cudaCheckError(cudaMallocManaged(&P, Mrows*Ncols*sizeof(float)));

  fillMatrix(M, Mrows, Mcols);
  printf("Matrix M: \n");
  printMatrix(M, Mrows, Mcols);

  fillMatrix(N, Nrows, Ncols);
  printf("Matrix N: \n");
  printMatrix(N, Nrows, Ncols);
 
  printf("\nProblem 1c.1: matrix multiplication using unified memory on CPU\n");
  elapsedTime = matmul(M, N, P, Mrows, Mcols, Nrows, Ncols);
  printf("Matrix M*N: \n");
  printMatrix(P, Mrows, Ncols);
	printf("Time to calculate results on CPU: %f ms.\n", elapsedTime);

  printf("\nProblem 1c.2: matrix multiplication using unified memory on GPU\n");
  elapsedTime = matmulDevice(M, N, P, Mrows, Mcols, Nrows, Ncols, unified);
  printf("Matrix M*N: \n");
  printMatrix(P, Mrows, Ncols);
	printf("Time to calculate results on GPU: %f ms.\n", elapsedTime);

  return 0;
}