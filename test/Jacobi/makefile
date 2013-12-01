

LNK_LIB = -lOpenCL

CC = g++

OPTS = -O3 -Wall

matrixMul: Jacobi.cpp 
	$(CC) $(OPTS) $@.cpp -o $@ $(LNK_LIB)


clean: 
	rm -f *~ *.o Jacobi