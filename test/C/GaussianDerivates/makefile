

LNK_LIB = -lOpenCL

INC_LIB = /usr/local/cuda-5.5/targets/x86_64-linux/include

CC = g++

OPTS = -O3 #-Wall

GaussianDerivates: GaussianDerivates.cpp 
	$(CC) $(OPTS) -I$(INC_LIB) $@.cpp -o $@ $(LNK_LIB)


clean: 
	rm -f *~ *.o GaussianDerivates