
EXEC = GaussianDerivates.exe


EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB = -lOpenCL -lrt

INC_LIB = include
CC = g++

OPTS = -O3 -Wall

EXECGPU = $(EXEC:.exe=GPU.exe)

DEFS = $(addprefix -D, $(DEF))

all: $(EXECGPU)


$(EXECGPU): $(EXECDEP)
	$(CC) $(OPTS) $(DEFS) -I$(INC_LIB) $(EXECDEP) -o $(@) $(LNK_LIB)

clean: 
	rm -f *~ *.o $(EXECGPU)
