#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "boilerplate.cpp"


using namespace std;

void
Jacobi(float* B, float* X1, float* X2, unsigned wA, unsigned wB)
{
  for (unsigned i = 1; i < wB; i++) {
    for (unsigned j = 1; j < wB; j++) {
      X2[i*wA + j] = -0.25 * (B[i * wB + j] -
			      (X1[(i-1) * wA + j] + X1[(i+1) * wA + j]) -
			      (X1[i * wA + (j-1)] + X1[i * wA + (j+1)]));
    }
  }
    
}

void
createB(float* B, unsigned wB, unsigned hB)
{
  float fwB = (float) wB;
  float fhB = (float) hB;
  float h_x = 1/(fwB);
  float h_y = 1/(fhB);
  float h_sq = 1/((fhB)*(fhB));
  for (unsigned i = 1; i < (hB); i++) {
    for (unsigned j = 1; j < (wB); j++) {
      B[i * wB + j] = -2*(M_PI * M_PI)*sin(M_PI*(j)*h_x) * sin(M_PI*(i)*h_y) * h_sq;
    }
  }
}

void
zeroMatrix(float* B, unsigned wB, unsigned hB)
{
  for (unsigned i = 0; i < (hB); i++) {
    for (unsigned j = 0; j < (wB); j++) {
      B[i * wB + j] = 0;
    }
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

void
printMat2(float* mat, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < (hMat); i++) {
    for (unsigned j = 0; j < (wMat); j++) {
      cout << mat[i * wMat + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void
printMat3(float* mat, unsigned size, unsigned wMat, unsigned hMat)
{
  unsigned count = 0;
  for (unsigned i = 1; i < (hMat) && count < size; i++) {
    for (unsigned j = 1; j < (wMat) && count < size; j++) {
      cout << mat[i * wMat + j] << " ";
      count++;
    }
    cout << endl;
  }
  cout << endl;
}

void
copyMat(float* mat1, float* mat2, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < hMat; i++) {
    for (unsigned j = 0; j < wMat; j++) {
      mat1[i * wMat + j] = mat2[i * wMat + j];
    }
  }
}


int main(int argc, char** argv)
{
  unsigned matsize;
  ParseCommandLine(argc, argv, &matsize, NULL, NULL);

  unsigned hA = matsize+2;
  unsigned hB = matsize+1;
  unsigned hC = matsize+2;
  unsigned wA = matsize+2;
  unsigned wB = matsize+1;
  unsigned wC = matsize+2;

  unsigned A_size = hA*wA;
  unsigned B_size = hB*wB;
  unsigned C_size = hC*wC;
  
  float* X1_mat = new float[A_size];
  float* B_mat = new float[B_size];
  float* X2_mat = new float[C_size];

  zeroMatrix(X1_mat, wA, hA);
  zeroMatrix(X2_mat, wC, hC);
  
  createB(B_mat, wB, hB);
  // printMat2(B_mat, wB, hB);

#if CPU
  timer.start();  
  Jacobi(B_mat, X1_mat, X2_mat, wA, wB);
  cout << "$Time " << timer.stop() << endl;  
#else 
  RunOCLJacobiForKernel(X2_mat, wC, hC,
			X1_mat, wA, hA,
			B_mat,  wB, hB,
			wB, wA
			);
#endif
  // for (unsigned i = 0; i < 10; i++) {
  //   for (unsigned j = 0; j < 10; j++) {
  //     cout << X2_mat[i * wA + j] << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;
#if PRINT
  printMat3(X2_mat, 100, wA, hA);
#endif
  // printMat(X2_mat, 100);

  free(X1_mat);
  free(B_mat);
  free(X2_mat);
  
}
