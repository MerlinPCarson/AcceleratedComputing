#include <stdlib.h>
#include <time.h>
#include <limits.h>


double vecMultCPU(double * a, double * b, int len){
	float elapsedTime;
  double result = 0;

  clock_t start = clock();

  // do vector multiplication
  for(int i=0; i<len; ++i){
    result += a[i] * b[i];
  }

  elapsedTime = ((float)(clock() - start)*1000)/CLOCKS_PER_SEC;

  // print process time
	printf("Time to calculate results on CPU: %f ms.\n", elapsedTime/1000);

  return result;
}

double vecMultOpenACC(double *restrict a, double *restrict b, int len){
	float elapsedTime;
  double result = 0;

  clock_t start = clock();

  // do vector multiplication
#pragma acc loop 
  for(int i=0; i<len; ++i){
    result += a[i] * b[i];
  }

  elapsedTime = ((float)(clock() - start)*1000)/CLOCKS_PER_SEC;

  // print process time
	printf("Time to calculate results using OpenACC: %f ms.\n", elapsedTime/1000);

  return result;
}

void fillVecs(double * a, double * b, int size){
  for(int i=0; i<size; ++i){
    a[i] = i;
    b[i] = i;
  }
}

void printUsage(){
  printf("\nUsage -- \n");
  printf("  dotProd <length of vectors>\n");
}

int isNumeric(char * str){
  int len = strlen(str);
  for(int i=0; i<len; ++i){
    if(!isdigit(str[i]))
        return 0; 
  }
  return 1;
}

int loadArguments(int argc, char * argv[], int * vecLength){

  if(argc != 2){
    printf("\nIncorrect number of arguments!\n\n");
  }
  else if (!isNumeric(argv[1])){
    printf("\nNon-numeric value found in command line arguments\n\n");
  }
  else if (atoi(argv[1]) <= 0 || atoi(argv[1]) >= INT_MAX){
    printf("\nVector length must be between 0 and %d\n", INT_MAX);
  }
  else{
    *vecLength = atoi(argv[1]);
    return 1;
  }

  printUsage();
  return 0;
}

int main(int argc, char * argv[])
{

  // variables for command line arguments
  int * vecLength = (int *)malloc(sizeof(int));

  // check number of and types of command line arguments
  if(!loadArguments(argc, argv, vecLength)){
    return 1;
  }

  // vector variables
  double *restrict a = (double*)malloc(*vecLength * sizeof(double));
  double *restrict b = (double*)malloc(*vecLength * sizeof(double));

  if(a == NULL || b == NULL){
    perror("malloc");
    exit(EXIT_FAILURE);
  }

  double result = 0;

  // fill vectors a and b with values
  fillVecs(a, b, *vecLength);

  printf("\nVector multiplication using CPU with %d elements:\n", *vecLength);

  result = vecMultCPU(a, b, *vecLength);
  printf("Result of vector multiplication on the CPU: %.2lf\n", result);

  printf("\nVector multiplication using OpenACC with %d elements\n", *vecLength);

  result = vecMultOpenACC(a, b, *vecLength);
  printf("Result of vector multiplication using OpenACC: %.2lf\n", result);

  // free memory
  free(a); free(b);
  free(vecLength);

  return 0;
}
