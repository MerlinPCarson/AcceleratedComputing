#include <stdio.h>
#include <cuda.h>

__global__
void vecAddKernel(float * a, float * b, float * c, int size)
{
  int i = blockDim.x*blockIdx.x+threadIdx.x;

  if (i<size) c[i] = a[i] + b[i];

}

extern "C" void vecAddGPU(float * h_a, float * h_b, float * h_c, int size)
{
  int n = size * sizeof(float);

  float * d_a, * d_b, * d_c;

  cudaMalloc((void**)&d_a, size);
  cudaMemcpy(d_a, h_a, n,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_b, size);
  cudaMemcpy(d_b, h_b, n,  cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_c, size);

  vecAddKernel<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, size);

  cudaMemcpy(h_c, d_b, n,  cudaMemcpyDeviceToHost);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void vecAddCPU(float * h_a, float * h_b, float * h_c, int size)
{
  int i;
  for(i=0;i<size;++i){
    h_c[i]=h_a[i]+h_b[i];
  }

}

void fillVecs(float * h_a, float * h_b, int size)
{
  int i;
  for(i=0;i<size;++i){
    h_a[i]=i;
    h_b[i]=i;
  }

}

void printVecs(float * h_a, float * h_b, float * h_c, int size)
{
  int i;
  for(i=0;i<size;++i){
    printf("a:%f b:%f, c:%f\n",h_a[i], h_b[i], h_c[i]);
  }

}

int main(void)
{
  int n = 1024;

  float * h_a = (float*)malloc(n * sizeof(float));
  float * h_b = (float*)malloc(n * sizeof(float));
  float * h_c = (float*)malloc(n * sizeof(float));

  fillVecs(h_a, h_b, n);
//  printVecs(h_a, h_b, h_c, n);

  printf("Vector addition with %d elements\n", n);

//  vecAddCPU(h_a, h_b, h_c, n);
//  printVecs(h_a, h_b, h_c, n);

  vecAddGPU(h_a, h_b, h_c, n);

  return 0;
}
