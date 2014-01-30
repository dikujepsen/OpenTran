

INCLUDE = /home/hpc/pr63so/di98raf/magma/magma-1.3.0
LIB = $(INCLUDE)/lib
TESTING = $(INCLUDE)/testing

INCLUDES=-I${CULA_INC_PATH}
LIBPATH32=-L${CULA_LIB_PATH_32}
LIBPATH64=-L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5

CUDAINC = -I/usr/local/cuda/include
CUDALINK = -L/usr/local/cuda/lib64


LUdecomp: 
	icc -O3 -DADD_ -DCUBLAS_GFORTRAN -DHAVE_CUBLAS -DGPUSHMEM=200 -I/usr/local/cuda/include -I$(INCLUDE)/testing -I$(INCLUDE)/include -I$(INCLUDE)/control -c testing_LUdecomp.cpp -o testing_LUdecomp.o

	icc -fPIC -Xlinker -zmuldefs -DGPUSHMEM=200  testing_LUdecomp.o  -o testing_LUdecomp $(TESTING)/libtest.a $(TESTING)/lin/liblapacktest.a -L$(LIB) -lmagma -lmagmablas -lmagma \
	-L/lrz/sys/intel/mkl110u1/lib/intel64 -L/usr/local/cuda/lib64 \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lcublas -lcudart -lm -openmp

systemSolve:
	icc -O3 -o systemSolve systemSolve.c $(CFLAGS) $(CUDAINC) $(INCLUDES) $(LIBPATH64) $(LIBS) $(CUDALINK)

