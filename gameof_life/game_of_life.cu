/* Compile with `gcc life.c`.
 * When CUDA-fied, compile with `nvcc life.cu`
 */

#include <cuda.h>
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep

#define WIDTH 60
#define HEIGHT 40


struct CudaBlockInfo{
  dim3 threadsPerBlock;
  dim3 blocksPerGrid;
};

// Function for checking for cuda errors
void cudaCheckError(cudaError_t err) {
  if(err!=cudaSuccess) {
    printf("Cuda failure %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
 }
}

const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
    {-1, 0},       {1, 0},
    {-1,-1},{0,-1},{1,-1}};

void fill_board(int *board, int width, int height) {
    int i;
    for (i=0; i<width*height; i++)
        board[i] = rand() % 2;
}

void print_board(int *board) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}

__global__
void stepKernel(int *current, int *next, int width, int height){

  int row = blockDim.y*blockIdx.y+threadIdx.y;
  int col = blockDim.x*blockIdx.x+threadIdx.x;
}

float stepGPU(int *h_current, int *h_next, int width, int height, CudaBlockInfo * blockInfo ) {
  // timing vars
	cudaEvent_t start, stop;
	float elapsedTime;

  size_t size = width * height * sizeof(int);
  int *d_current, *d_next;

  // alocate memory and move vectors to GPU
  cudaCheckError(cudaMalloc((void**)&d_current, size));
  cudaCheckError(cudaMemcpy(d_current, h_current, size,  cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc((void**)&d_next, size));
  cudaCheckError(cudaMemcpy(d_next, h_next, size,  cudaMemcpyHostToDevice));

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // peform operation on GPU
  stepKernel<<<blockInfo->blocksPerGrid, blockInfo->threadsPerBlock>>>(d_current, d_next, width, height);
  // copy results back to CPU
  cudaCheckError(cudaMemcpy(h_next, d_next, size,  cudaMemcpyDeviceToHost));
  
  // end timings
	cudaEventRecord(stop, 0);     
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

  // free GPU memory
  cudaFree(d_current); cudaFree(d_next);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  
  return elapsedTime;
}

float step(int *current, int *next, int width, int height) {
    // timing vars
    cudaEvent_t start, stop;
    float elapsedTime;
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // start timings
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // write the next board state
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
 
         

            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + width) % width;
                ny = (y + offsets[i][1] + height) % height;
                if (current[ny * width + nx]) {
                    num_neighbors++;
                }
            }

            // apply the Game of Life rules to this cell
            next[y * width + x] = 0;
            if ((current[y * width + x] && num_neighbors==2) ||
                    num_neighbors==3) {
                next[y * width + x] = 1;
            }
        }
    }

    // end timings
	  cudaEventRecord(stop, 0);     
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&elapsedTime, start, stop);

    return elapsedTime;
}

int main(int argc, const char *argv[]) {

    bool on_gpu = true;
    CudaBlockInfo * blockInfo = (CudaBlockInfo *)malloc(sizeof(CudaBlockInfo));

    // parse the width and height command line arguments, if provided
    int width, height, iters, out;
    if (argc < 3) {
        printf("usage: life iterations 1=print"); 
        exit(1);
    }
    iters = atoi(argv[1]);
    out = atoi(argv[2]);
    if (argc == 5) {
        width = atoi(argv[3]);
        height = atoi(argv[4]);
        printf("Running %d iterations at %d by %d pixels.\n", iters, width, height);
    } else {
        width = WIDTH;
        height = HEIGHT;
    }

    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    float procTime;
    // The two boards 
    int *current, *next, many=0;

    size_t board_size = sizeof(int) * width * height;
    current = (int *) malloc(board_size); // same as: int current[width * height];
    next = (int *) malloc(board_size);    // same as: int next[width *height];
 

    // Initialize the global "current".
    fill_board(current, width, height);


    while (many<iters) {
        many++;
        if (out==1)
            print_board(current);

        if (on_gpu == false){
          printf("Processing on CPU\n");
          //evaluate the `current` board, writing the next generation into `next`.
          procTime = step(current, next, width, height);
          // print process time
          printf("Time to calculate results on CPU: %f ms.\n", procTime);
          // Copy the next state, that step() just wrote into, to current state
          memcpy(current, next, board_size);
        }
        else{
          printf("Processing on GPU\n");
          // copy the `next` to CPU and into `current` to be ready to repeat the process
          procTime = stepGPU(current, next, width, height, blockInfo);
          // print process time
          printf("Time to calculate results on GPU: %f ms.\n", procTime);
          memcpy(current, next, board_size);
        }


        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out==1)
            nanosleep(&delay, &remaining);
    }

    free(blockInfo);
    free(current);
    free(next);

    return 0;
}
