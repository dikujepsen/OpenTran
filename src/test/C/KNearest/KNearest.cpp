


#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <assert.h>
#include <float.h>

#include <string.h>
#include "boilerplate.cpp"

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

#if PRINT
  printMat(dist_matrix, 100);
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
