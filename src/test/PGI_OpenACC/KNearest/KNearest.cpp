


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

#include "../../../utils/Stopwatch.cpp"
Stopwatch timer;


using namespace std;
void
printMat(float* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    cout << mat[i] << " ";
    if (i % 10 == 0) {
      cout << endl;
    }
  }
  cout << endl;
}


void knearest(unsigned NTEST, unsigned NTRAIN, unsigned dim, float *__restrict__ test_patterns, float *__restrict__ train_patterns, float *__restrict__ dist_matrix) {

#ifndef CPU
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
  unsigned NTRAIN;
  unsigned dim;
  ParseCommandLine(argc, argv, &NTRAIN, NULL, &dim);
  unsigned NTEST = 3*NTRAIN;
  // size_t NTRAIN = 4096;
  // size_t NTEST = 3*NTRAIN;
  // size_t dim = 16;
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
  timer.start();  
  
#ifndef CPU
#ifdef READBACK
#pragma acc kernels loop copyin(test_patterns[0:NTEST*dim], train_patterns[0:NTRAIN*dim])  copyout(dist_matrix[0:NTEST*NTRAIN] ) independent
#else
#pragma acc kernels loop copyin(test_patterns[0:NTEST*dim], train_patterns[0:NTRAIN*dim])  local(dist_matrix[0:NTEST*NTRAIN] ) independent
#endif
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
  cout << "$Time " << timer.stop() << endl;  
   
#if PRINT
  printMat(dist_matrix, 100);
#endif

  
    
  return 0;
}
