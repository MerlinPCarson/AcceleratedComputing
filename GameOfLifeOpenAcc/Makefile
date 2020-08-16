PGCC=/pkgs/pgi/19.10/linux86-64/19.10/bin/pgcc 
# For GPU
FLAGS=-Minfo=accel -acc
# For Multi-Core CPU
#FLAGS=-Minfo=accel -acc -ta=multicore
# For sequential
#FLAGS=
EXE=gol
SOURCE=gol_openAcc.c

all: 
	$(PGCC) $(FLAGS) -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE)
