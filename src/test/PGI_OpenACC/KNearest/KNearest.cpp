


#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <assert.h>
#include <float.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <openacc.h>

#include "../../../src/utils/Stopwatch.cpp"
Stopwatch timer;

long get_system_time_in_microseconds(void){
	struct timeval tempo;
	gettimeofday(&tempo, NULL);
	return tempo.tv_sec * 1000000 + tempo.tv_usec;	
}

#define TIMING

#ifdef TIMING
#define DEFINE_TIMER(num) long start_time##num = 0; double elapsed_time##num = 0.0f;
#define DECLARE_TIMER(num) extern long start_time##num; extern double elapsed_time##num;
#define START_TIMER(num) start_time##num = get_system_time_in_microseconds();
#define STOP_TIMER(num) elapsed_time##num = (((double)get_system_time_in_microseconds())-((double)start_time##num));
#define GET_TIME(num) (double)(1.0*elapsed_time##num / 1000000.0)
#else
#define DEFINE_TIMER(num) 
#define DECLARE_TIMER(num)
#define START_TIMER(num) 
#define STOP_TIMER(num) 
#define GET_TIME(num)
#endif

DEFINE_TIMER(1); 



void knearest(unsigned NTEST, unsigned NTRAIN, unsigned dim, float *__restrict__ test_patterns, float *__restrict__ train_patterns, float *__restrict__ dist_matrix) {

#if 1
#pragma acc kernels loop copyin(test_patterns[0:NTEST*dim], train_patterns[0:NTRAIN*dim])  copyout(dist_matrix[0:NTEST*NTRAIN] ) independent
#endif
  for (unsigned i = 0; i < NTEST; i++) {
    for (unsigned j = 0; j < NTRAIN; j++) {
      float d,tmp;
      d = 0.0;
      // for each feature
      for (unsigned k = 0; k < dim; k++) {
	tmp = test_patterns[i*dim+k]-train_patterns[j*dim+k];
	d += tmp*tmp;
      }
      dist_matrix[j*NTEST + i] = d;
    }
  }
}

int main( int argc, char* argv[] )
{
  // problem parameters
  size_t NTRAIN = 4096;
  size_t NTEST = 3*NTRAIN;
  size_t dim = 16;
  // size_t NTRAIN = 16;
  // size_t NTEST = 3*NTRAIN;
  // size_t dim = 16;
  acc_init( acc_device_nvidia );

  size_t i,j,k;
  
  // training and test patterns
  float *train_patterns = (float*)malloc(NTRAIN*dim*sizeof(float));
  float *test_patterns = (float*)malloc(NTEST*dim*sizeof(float));
  // result
  float *dist_matrix = (float*)malloc(NTRAIN*NTEST*sizeof(float));
  // initialize with some values ...
  for (i=0;i<NTRAIN;i++){
    for (k=0;k<dim;k++){
      train_patterns[i*dim + k] = (float)sin(i);
    }
  }
  for (i=0;i<NTEST;i++){
    for (k=0;k<dim;k++){
      test_patterns[i*dim + k] = (float)cos(i);
    }
  }
  
#if 1
  START_TIMER(1); 

#if 1
#pragma acc kernels loop copyin(test_patterns[0:NTEST*dim], train_patterns[0:NTRAIN*dim])  copyout(dist_matrix[0:NTEST*NTRAIN] ) independent
#endif
  for (unsigned i = 0; i < NTEST; i++) {
    for (unsigned j = 0; j < NTRAIN; j++) {
      float d,tmp;
      d = 0.0;
      // for each feature
      for (unsigned k = 0; k < dim; k++) {
	tmp = test_patterns[i*dim+k]-train_patterns[j*dim+k];
	d += tmp*tmp;
      }
      dist_matrix[j*NTEST + i] = d;
    }
  }
  // knearest(NTEST, NTRAIN, dim, test_patterns, train_patterns,
  // 	   dist_matrix);
   
   
  STOP_TIMER(1);
  
  printf("\nElapsed time=%f\n", GET_TIME(1));
  
#else
#endif

  // print matrix
  for (i=0;i<10;i++){    
    for (j=0;j<10;j++){
      printf("%f ", dist_matrix[j*NTEST+i]);
    }
    printf("\n");
  }

  // final timing results
    
  return 0;
}
