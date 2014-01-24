

EXECDEP = $(EXEC:.exe=.cpp)

LNK_LIB = -acc -Minfo=accel -fast

INC_LIB = /usr/include/x86_64-linux-gnu -Msafeptr

CC = pgCC

OPTS = -O3 

$(EXEC): $(EXECDEP)
	$(CC) $(OPTS) -I$(INC_LIB) $(@:.exe=.cpp) -o $@ $(LNK_LIB)


clean: 
	rm -f *~ *.o $(EXEC)