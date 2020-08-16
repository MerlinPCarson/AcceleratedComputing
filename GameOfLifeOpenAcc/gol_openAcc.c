#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep

#define WIDTH 60
#define HEIGHT 40


const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
    {-1, 0},       {1, 0},
    {-1,-1},{0,-1},{1,-1}};

void fill_board(int * restrict board, int width, int height) {
    int i;
    for (i=0; i<width*height; i++)
        board[i] = rand() % 2;
}

void print_board(int * restrict board) {
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


float step(int * restrict current, int * restrict next, int width, int height) {
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
    int board_len = width * height;

  // timing vars
  clock_t start = clock();
  //float totalTime = 0.0;

#pragma acc data copyin(current[0:board_len], offsets[0:8]) copyout(next[0:board_len])
#pragma acc parallel loops
    for (y=0; y<height; y++) {
        for (x=0; x<width; x++) {

            // count this cell's alive neighbors
            num_neighbors = 0;
#pragma acc loops reduction(+:num_neighbors) 
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

  return ((float)(clock() - start)*1000)/CLOCKS_PER_SEC;
}

int main(int argc, const char *argv[]) {

    int on_gpu = 0;

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
    // The two boards 
    int * restrict current, * restrict next, many=0;

    size_t board_size = sizeof(int) * width * height;
    current = (int *) malloc(board_size); // same as: int current[width * height];
    next = (int *) malloc(board_size);    // same as: int next[width *height];
 

    // Initialize the global "current".
    fill_board(current, width, height);

    float totalTime = 0.0;

    while (many<iters) {
        many++;
        if (out==1)
            print_board(current);

        //evaluate the `current` board, writing the next generation into `next`.
        totalTime += step(current, next, width, height);
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, board_size);


        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out==1)
            nanosleep(&delay, &remaining);
    }
    
    printf("average of execution time of total process is %f\n", totalTime/iters);

    return 0;
}
