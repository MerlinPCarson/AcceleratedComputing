/* Compile with `gcc life.c`.
 * When CUDA-fied, compile with `nvcc life.cu`
 */

#include <cuda.h>
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep
#include <device_types.h>

#define WIDTH 60
#define HEIGHT 40
#define BLOCKSIZE 16
#define TILESIZE 16


struct CudaBlockInfo{
  dim3 threadsPerBlock;
  dim3 threadsPerTile;
  dim3 blocksPerGrid;
  dim3 tilesPerGrid;
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

void print_board(int *board, int width, int height) {
    int x, y;
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {
            char c = board[y * width + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}


void print_boards(int *boardCPU, int *boardGPU, int width, int height) {
    int x, y;
    for (y=0; y<height; ++y) {
        for (x=0; x<2*width; ++x) {
            if(x<width){
              char c = boardCPU[y * width + x] ? '#':' ';
              printf("%c", c);
            }
            else{
              if(x==width){
                printf("\t");
              }
              char c = boardGPU[y%width * width + x%width] ? '#':' ';
              printf("%c", c);
            }
        }
        printf("\n");
    }
    for(y=0; y<2; ++y){
      for(x=0; x<width+3; ++x){
        if (y==1 && x==0){
          printf("\t");
        }
        if((x<((width/2)-2)) || (x>((width/2)+2))){
          printf("-");
        }
        else if(x==(width/2)){
          if(y==0){
            printf("CPU");
          }
          else{
            printf("GPU");
          }

        }
      }
    }
    printf("\n");
}

__global__
void stepTileKernel(int *current, int *next, int width, int height){

  int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                             {-1, 0},       {1, 0},
                             {-1,-1},{0,-1},{1,-1}};

  __shared__ int c_ds[TILESIZE+2][TILESIZE+2];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILESIZE + ty;
  int col = bx * TILESIZE + tx;

  //printf("(%d, %d), (%d, %d) \n", col, row, bx, by);
  //printf("(%d, %d) \n", tx, ty);

  int nx, ny;
  int num_neighbors = 0;

  // load cells data
  if((row < height) && (col < width)){
    c_ds[ty+1][tx+1] = current[row * width + col];
  }

//  tile boundry 
  if (ty == 0){
    if (tx == 0){
      nx = (col - 1 + width) % width;
      ny = (row - 1 + height) % height;
//      printf("Filling (%d,%d) with (%d, %d)\n",row, col, ny,nx);
      c_ds[ty][tx] = current[ny * width + nx];
    }
    else if(tx == (TILESIZE-1)){
      nx = (col + 1 + width) % width;
      ny = (row - 1 + height) % height;
//      printf("Filling (%d,%d) with (%d, %d)\n",row, col+2, ny,nx);
      c_ds[ty][tx+2] = current[ny * width + nx];
    }
    ny = (row - 1 + height) % height;
    c_ds[ty][tx+1] = current[ny * width + col];
  }
  else if(ty == (TILESIZE-1)){
    if (tx == 0){
      nx = (col - 1 + width) % width;
      ny = (row + 1 + height) % height;
//      printf("Filling (%d,%d) with (%d, %d)\n",row+2, col, ny,nx);
      c_ds[ty+2][tx] = current[ny * width + nx];
    }
    else if(tx == (TILESIZE-1)){
      nx = (col + 1 + width) % width;
      ny = (row + 1 + height) % height;
//      printf("Filling (%d,%d) with (%d, %d)\n",row+2, col+2, ny,nx);
      c_ds[ty+2][tx+2] = current[ny * width + nx];
    }
    ny = (row + 1 + height) % height;
    c_ds[ty+2][tx+1] = current[ny * width + col];
  }
  if (tx == 0){
    nx = (col - 1 + width) % width;
//      printf("Filling (%d,%d) with (%d, %d)\n",row+1, col, row,nx);
    c_ds[ty+1][tx] = current[row * width + nx];
  }
  else if(tx == (TILESIZE-1)){
    nx = (col + 1 + width) % width;
//    printf("Filling (%d,%d) with (%d, %d)\n",row+1, col+2, row,nx);
    c_ds[ty+1][tx+2] = current[row * width + nx];
  }

  __syncthreads();

  if ((row < height) && (col < width)){
      // count this cell's alive neighbors
      for (int i=0; i<8; i++) {
          if (c_ds[ty+1+offsets[i][0]][tx+1+offsets[i][1]]) {
              num_neighbors++;
          }
        }

      // apply the Game of Life rules to this cell
      next[row * width + col] = 0;
      if ((current[row * width + col] && num_neighbors==2) ||
          num_neighbors==3) {
        next[row * width + col] = 1;
      }

  }
//  if(tx == 0 && ty == 0){
//    for (int y=0; y<TILESIZE+2; y++) {
//        for (int x=0; x<TILESIZE+2; x++) {
//          if(x==0 || x == TILESIZE+1 || y==0 || y==TILESIZE+1){
//            printf(" ");
//          }
//          else{
//            char c = c_ds[y][x] ? '#':' ';
//            printf("%c", c);
//          }
//        }
//        printf("\n");
//    }
//    printf("-----\n");
//  }
//  if(tx == 0 && ty == 0){
//    for (int y=0; y<TILESIZE+2; y++) {
//        for (int x=0; x<TILESIZE+2; x++) {
//            char c = c_ds[y][x] ? '#':' ';
//            printf("%c", c);
//        }
//        printf("\n");
//    }
//    printf("-----\n");
//  }

  __syncthreads();

}

__global__
void stepKernel(int *current, int *next, int width, int height){

  int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                             {-1, 0},       {1, 0},
                             {-1,-1},{0,-1},{1,-1}};

  int row = blockDim.y*blockIdx.y+threadIdx.y;
  int col = blockDim.x*blockIdx.x+threadIdx.x;

//    printf("(%d, %d) \n", col, row);

  int nx, ny;
  int num_neighbors = 0;

  if ((row < height) && (col < width)){
      // count this cell's alive neighbors
      for (int i=0; i<8; i++) {
          // To make the board torroidal, we use modular arithmetic to
          // wrap neighbor coordinates around to the other side of the
          // board if they fall off.
          nx = (col + offsets[i][0] + width) % width;
          ny = (row + offsets[i][1] + height) % height;
          if (current[ny * width + nx]) {
              num_neighbors++;
          }
        }

      // apply the Game of Life rules to this cell
      next[row * width + col] = 0;
      if ((current[row * width + col] && num_neighbors==2) ||
          num_neighbors==3) {
          next[row * width + col] = 1;
      }

  }
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

  // start timings
	cudaEventCreate(&start);    
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  // peform operation on GPU
  printf("%d,%d,%d:%d,%d,%d\n", blockInfo->threadsPerBlock.x, blockInfo->threadsPerBlock.y, blockInfo->threadsPerBlock.z, blockInfo->blocksPerGrid.x, blockInfo->blocksPerGrid.y, blockInfo->blocksPerGrid.z);
  //stepKernel<<<blockInfo->blocksPerGrid, blockInfo->threadsPerBlock>>>(d_current, d_next, width, height);
  stepTileKernel<<<blockInfo->tilesPerGrid, blockInfo->threadsPerTile>>>(d_current, d_next, width, height);
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


    // parse the width and height command line arguments, if provided
    int width, height, numIters, out;
    if (argc < 3) {
        printf("usage: life iterations 1=print"); 
        exit(1);
    }
    numIters = atoi(argv[1]);
    out = atoi(argv[2]);
    if (argc == 5) {
        width = atoi(argv[3]);
        height = atoi(argv[4]);
        printf("Running %d iterations at %d by %d pixels.\n", numIters, width, height);
    } else {
        width = WIDTH;
        height = HEIGHT;
    }

    // GPU vars
    //bool on_gpu = true;
    CudaBlockInfo * blockInfo = (CudaBlockInfo *)malloc(sizeof(CudaBlockInfo));
    dim3 blocksPerGrid(ceil((float)width/BLOCKSIZE), ceil((float)height/BLOCKSIZE), 1);
    dim3 tilesPerGrid(ceil((float)width/TILESIZE), ceil((float)height/TILESIZE), 1);
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 threadsPerTile(TILESIZE, TILESIZE, 1);
    blockInfo->threadsPerTile = threadsPerTile;
    blockInfo->threadsPerBlock = threadsPerBlock;
    blockInfo->blocksPerGrid = blocksPerGrid;
    blockInfo->tilesPerGrid = tilesPerGrid;
    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;

    float procTime = 0;
    float totalProcTimeCPU = 0;
    float totalProcTimeGPU = 0;

    // The two boards 
    int currIter = 0;
    int *start, *current, *next;
    int *currentGPU, *nextGPU;

    size_t board_size = sizeof(int) * width * height;
    start = (int *) malloc(board_size); // same as: int current[width * height];
    current = (int *) malloc(board_size); // same as: int current[width * height];
    next = (int *) malloc(board_size);    // same as: int next[width *height];
    currentGPU = (int *) malloc(board_size); // same as: int current[width * height];
    nextGPU = (int *) malloc(board_size);    // same as: int next[width *height];
 
    printf("Initializing boards\n"); 
    fill_board(start, width, height);
    memcpy(current, start, board_size);
    memcpy(currentGPU, start, board_size);


    // Run on CPU
    while (currIter<numIters) {
        ++currIter;
        if (out==1)
            print_boards(current, currentGPU, width, height);

        //evaluate the `current` board, writing the next generation into `next`.
        procTime = step(current, next, width, height);
        totalProcTimeCPU += procTime;

        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, board_size);

        // print process time
        //printf("Time to calculate results on CPU: %f ms.\n", procTime);

        // copy the `next` to CPU and into `current` to be ready to repeat the process
        procTime = stepGPU(currentGPU, nextGPU, width, height, blockInfo);
        totalProcTimeGPU += procTime;

        // Copy the next state, that step() just wrote into, to current state
        memcpy(currentGPU, nextGPU, board_size);

        // print process time
        //printf("Time to calculate results on GPU: %f ms.\n", procTime);

        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out==1)
            nanosleep(&delay, &remaining);
    }


    printf("Average processing time on CPU is %f ms.\n", (totalProcTimeCPU/numIters));
    printf("Average processing time on GPU is %f ms.\n", (totalProcTimeGPU/numIters));

    free(blockInfo);
    free(start);
    free(current);
    free(next);
    free(currentGPU);
    free(nextGPU);

    return 0;
}
