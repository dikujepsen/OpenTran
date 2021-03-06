#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "boilerplate.cpp"
#include "helper.cpp"

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



int main(int argc, char** argv)
{
  unsigned matsize = 4096;
 
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
  float* X2_mat_cpu = new float[C_size];
  float* X2_mat = new float[C_size];

  zeroMatrix(X1_mat, wA, hA);
  zeroMatrix(X2_mat_cpu, wC, hC);
  zeroMatrix(X2_mat, wC, hC);

  createB(B_mat, wB, hB);

  Stopwatch timer;
  timer.start();

  Jacobi(B_mat, X1_mat, X2_mat_cpu, wA, wB);
  cout << "$Sequential_time " << timer.stop() << endl;  
  
  OCLJacobiTask ocl_task;
  ocl_task.RunOCLJacobiForKernel(
			B_mat,  wB, hB,
			X1_mat, wA, hA,
			X2_mat, wC, hC,
			"gpu",
			wB
			);

  

  free(X1_mat);
  free(B_mat);
  free(X2_mat);
  
}
