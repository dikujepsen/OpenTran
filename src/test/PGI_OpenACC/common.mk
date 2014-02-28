

EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB =  -fast -Minline=levels:3 -acc -Minfo=accel 

INC_LIB = /usr/include/x86_64-linux-gnu -Msafeptr -ta=kepler

#CC = /home/xhz206/PGI_Compiler/linux86-64/13.10/bin/pgCC
#CC = /home/jacob/PGI_Compiler/linux86-64/13.10/bin/pgCC
CC = pgCC

OPTS = -O3 

EXECCPU = $(EXEC:.exe=CPU.exe)
EXECGPU += $(EXEC:.exe=GPU.exe)

DEFS = $(addprefix -D, $(DEF))

all: $(EXECCPU) $(EXECGPU)

$(EXECCPU): $(EXECDEP)
	$(CC) $(OPTS) $(DEFS) -DCPU -I$(INC_LIB) $(EXECDEP) -o $(@) $(LNK_LIB)

$(EXECGPU): $(EXECDEP)
	$(CC) $(OPTS) $(DEFS) -I$(INC_LIB) $(EXECDEP) -o $(@) $(LNK_LIB)

clean: 
	rm -f *~ *.o $(EXECCPU) $(EXECGPU)