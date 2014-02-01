

EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB = -acc -Minfo=accel -fast

INC_LIB = /usr/include/x86_64-linux-gnu -Msafeptr

CC = pgCC

OPTS = -O3 

$(EXEC): $(EXECDEP)
	$(CC) $(OPTS) -DCPU -I$(INC_LIB) $(@:.exe=.cpp) -o $(@:.exe=CPU.exe) $(LNK_LIB)
	$(CC) $(OPTS) -I$(INC_LIB) $(@:.exe=.cpp) -o $(@:.exe=GPU.exe) $(LNK_LIB)

clean: 
	rm -f *~ *.o $(EXEC)