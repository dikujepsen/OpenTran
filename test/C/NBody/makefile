

LNK_LIB = -lOpenCL -lrt

CC = g++

OPTS = -O3 -Wall

NBody: NBody.cpp 
	$(CC) $(OPTS) $@.cpp -o $@ $(LNK_LIB)


clean: 
	rm -f *~ *.o NBody