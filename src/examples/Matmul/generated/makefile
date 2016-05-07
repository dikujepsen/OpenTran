
EXEC = MatMul.exe


EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB = -lOpenCL -lrt

#INC_LIB = /usr/local/cuda-5.5/targets/x86_64-linux/include
#INC_LIB = /opt/intel/opencl-1.2-3.2.1.16712/include
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
