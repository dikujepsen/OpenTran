


#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <assert.h>
#include <float.h>

#include <string.h>
#include "boilerplate.cpp"
#include "../../../utils/helper.cpp"


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

void
zeroMatrix(float* B, unsigned b_size)
{
  for (unsigned i = 0; i < b_size; i++) {
      B[i] = 0;
  }
}

int main( int argc, char* argv[] )
{
  // problem parameters
  unsigned NTRAIN;
  unsigned dim;
  ParseCommandLine(argc, argv, &NTRAIN, NULL, &dim, NULL);
  unsigned NTEST = 3*NTRAIN;
  // size_t NTRAIN = 16;
  // size_t NTEST = 3*NTRAIN;
  // size_t dim = 16;

  size_t i,j,k;
  
  // training and test patterns
  float *train_patterns = (float*)malloc(NTRAIN*dim*sizeof(float));
  float *test_patterns = (float*)malloc(NTEST*dim*sizeof(float));
  // result
  unsigned dist_matrix_size = NTRAIN*NTEST;
  float *dist_matrix_cpu = (float*)malloc(dist_matrix_size*sizeof(float));
  float *dist_matrix = (float*)malloc(dist_matrix_size*sizeof(float));
  zeroMatrix(dist_matrix_cpu, dist_matrix_size);
  zeroMatrix(dist_matrix, dist_matrix_size);

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
      dist_matrix_cpu[j*NTEST + i] = d;
    }
  }
  cout << "$Time " << timer.stop() << endl;  
  
  RunOCLKNearestForKernel(
	NTEST, NTRAIN, dim,
	dist_matrix, NTEST, NTRAIN,
	"gpu", test_patterns, dim, NTEST, train_patterns,
	dim, NTRAIN);


//#if PRINT
//  printMat(dist_matrix, 100);
//#endif

  helper::check_matrices(dist_matrix_cpu, dist_matrix, dist_matrix_size);

    
  return 0;
}
