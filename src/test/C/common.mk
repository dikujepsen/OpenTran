

EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB = -lOpenCL -lrt

# INC_LIB = /usr/local/cuda-5.5/targets/x86_64-linux/include
INC_LIB = /opt/intel/opencl-1.2-3.2.1.16712/include
CC = g++

OPTS = -O3 -Wall

EXECCPU = $(EXEC:.exe=CPU.exe)
EXECGPU += $(EXEC:.exe=GPU.exe)

DEFS = $(addprefix -D, $(DEF))

all: $(EXECGPU) # $(EXECCPU) $(EXECGPU)

#$(EXECCPU): $(EXECDEP)
#	$(CC) $(OPTS) $(DEFS) -DCPU -I$(INC_LIB) $(EXECDEP) -o $(@) $(LNK_LIB)

$(EXECGPU): $(EXECDEP)
	$(CC) $(OPTS) $(DEFS) -I$(INC_LIB) $(EXECDEP) -o $(@) $(LNK_LIB)

clean: 
	rm -f *~ *.o $(EXECCPU) $(EXECGPU)