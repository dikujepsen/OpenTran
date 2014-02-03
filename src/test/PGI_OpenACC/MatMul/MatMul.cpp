#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "../../../utils/Stopwatch.cpp"
Stopwatch timer;
#include <openacc.h>

using namespace std;

void
matmul(float* A, float* B, float* C, unsigned hA, unsigned wA, unsigned wB)
{
  #ifndef CPU
#pragma acc kernels loop copyin(A[0:hA*wA], B[0:wB*wA])  local(C[0:wB*hA] ) independent
#endif
  for (unsigned i = 0; i < hA; i++) {
    for (unsigned j = 0; j < wB; j++) {
      float sum = 0;
      for (unsigned k = 0; k < wA; k++) {
	sum += A[i * wA + k] * B[k * wB + j];
      }
      C[i * wB + j] = sum;
    }
  }
}

void
randMat(float* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = (float)((rand() % 10)/10.0);
  }
}

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


// #define matsize 3584
//#define matsize 9216

int main(int argc, char** argv)
{
  unsigned matsize;
  ParseCommandLine(argc, argv, &matsize, NULL, NULL);
  unsigned hA = matsize;
  unsigned hB = matsize;
  unsigned hC = matsize;
  unsigned wA = matsize;
  unsigned wB = matsize;
  unsigned wC = matsize;

  unsigned A_size = hA*wA;
  unsigned B_size = hB*wB;
  unsigned C_size = hC*wC;
  
  float* A_mat = new float[A_size];
  float* B_mat = new float[B_size];
  float* C_mat = new float[C_size];

  srand(2013);

  randMat(A_mat,A_size);
  randMat(B_mat,B_size);
  randMat(C_mat,C_size);
  acc_init( acc_device_nvidia );

  timer.start();
  matmul(A_mat, B_mat, C_mat, hA, wA, wB);
  cout << timer.stop() << endl;

  // printMat(C_mat, 10);

  free(A_mat);
  free(B_mat);
  free(C_mat);
  
}
