#include <cstdlib>
#include <cstdio>
#include <iostream>

using namespace std;

void
matmul(float* B, float* X1, float* X2, unsigned hA, unsigned wA)
{
  for (unsigned i = 1; i < (hA+1); ++i) {
    for (unsigned j = 1; j < (wB+1); ++j) {
      X2[i*wA + j] = B[(i-1) * wA + (j-1)] - 
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


#define matsize 8

int main(int argc, char** argv)
{
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
  hst_ptrA_dim1 = wA;
  hst_ptrA_dim2 = hA;
  float* B_mat = new float[B_size];
  float* C_mat = new float[C_size];

  srand(2013);

  randMat(A_mat,A_size);
  randMat(B_mat,B_size);
  randMat(C_mat,C_size);
#define GPU 1
#if GPU
  RunOCLMatmulKernel(A_mat,wA,hA,
		     B_mat,wB,hB,
		     C_mat,wC,hC);
#else
  matmul(A_mat, B_mat, C_mat, hA, wA, wB);
#endif
  printMat(C_mat,C_size);

  free(A_mat);
  free(B_mat);
  free(C_mat);
  
}
