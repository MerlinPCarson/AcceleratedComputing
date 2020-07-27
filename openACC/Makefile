PGCC=/pkgs/pgi/19.10/linux86-64/19.10/bin/pgcc 
FLAGS=-Minfo=accel -acc
EXE=dotProd
SOURCE=dotProd.c

all: 
	$(PGCC) $(FLAGS) -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE)
