#include <stdlib.h>
#include <random>


void matmul(float * M, float * N, float * P, int Mrows, int Mcols, int Nrows, int Ncols ){

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

  int Mrows = 9;
  int Mcols = 7;
  int Nrows = Mcols;
  int Ncols = 11;

  float * M = (float*)malloc(Mrows*Mcols*sizeof(float));
  float * N = (float*)malloc(Mrows*Mcols*sizeof(float));

  float * P = (float*)calloc(Mrows*Ncols, sizeof(float));

  fillMatrix(M, Mrows, Mcols);
  printf("Matrix M: \n");
  printMatrix(M, Mrows, Mcols);

  fillMatrix(N, Nrows, Ncols);
  printf("Matrix N: \n");
  printMatrix(N, Nrows, Ncols);
  
  matmul(M, N, P, Mrows, Mcols, Nrows, Ncols);
  printf("Matrix M*N: \n");
  printMatrix(P, Mrows, Ncols);
}
