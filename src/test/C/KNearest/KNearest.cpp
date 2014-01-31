


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
#include "boilerplate.cpp"

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
 

int main( int argc, char* argv[] )
{
  // problem parameters
  unsigned NTRAIN;
  unsigned dim;
  ParseCommandLine(argc, argv, &NTRAIN, NULL, &dim);
  unsigned NTEST = 3*NTRAIN;
  // size_t NTRAIN = 16;
  // size_t NTEST = 3*NTRAIN;
  // size_t dim = 16;

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
  
#if CPU
  timer.start();  
  float d,tmp;
  for (i=0;i<NTEST;i++) {
    // compute distances to all training patterns
    for (j=0;j<NTRAIN;j++) {
      d = 0.0;
      // for each feature
      for (k=0;k<dim;k++) {
	tmp = test_patterns[i*dim+k]-train_patterns[j*dim+k];
	d += tmp*tmp;
      }
      dist_matrix[j*NTEST + i] = d;
    }
  }
  cout << "$Time " << timer.stop() << endl;  
  
#else
  RunOCLKNearestForKernel(dim, test_patterns, dim, 
			  NTEST, dist_matrix, NTEST, NTRAIN,
			  train_patterns, dim, NTRAIN,
			  NTEST, NTRAIN);

#endif

  // print matrix
  // for (i=0;i<10;i++){    
  //   for (j=0;j<10;j++){
  //     printf("%f ", dist_matrix[j*NTEST+i]);
  //   }
  //   printf("\n");
  // }

    
  return 0;
}
