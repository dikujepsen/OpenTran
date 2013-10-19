

LNK_LIB = -lOpenCL

CC = g++

OPTS = -O3 -Wall

matrixMul: matrixMul.cpp oclMatmul.cpp
	$(CC) $(OPTS) $@.cpp -o $@ $(LNK_LIB)

