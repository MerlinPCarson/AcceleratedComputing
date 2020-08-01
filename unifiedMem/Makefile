PGCC=/pkgs/pgi/19.10/linux86-64/19.10/bin/pgcc 
NVCC=nvcc 
FLAGS=-Minfo=accel -acc
EXE=matmul
SOURCE=main.cpp

all: 
#	$(PGCC) $(FLAGS) -o $(EXE) $(SOURCE) 
	$(NVCC) -x cu -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE)
