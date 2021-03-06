#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "boilerplate.cpp"
#include "helper.cpp"

using namespace std;

void
matmul(float* A, float* B, float* C, unsigned hA, unsigned wA, unsigned wB)
{
  for (unsigned i = 0; i < hA; i++)
  {
    for (unsigned j = 0; j < wB; j++)
    {
      float sum = 0;
      for (unsigned k = 0; k < wA; k++)
      {
        sum += A[i * wA + k] * B[k * wB + j];
      }
      C[i * wB + j] = sum;
    }
  }
}

void
randMat(float* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i)
  {
    mat[i] = (float)((rand() % 10)/10.0);
  }
}


int main(int argc, char** argv)
{
  unsigned matsize = 1024;

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
  float* C_mat_cpu = new float[C_size];
  srand(2013);

  randMat(A_mat,A_size);
  randMat(B_mat,B_size);
  randMat(C_mat,C_size);


  Stopwatch timer;
  timer.start();
  matmul(A_mat, B_mat, C_mat_cpu, hA, wA, wB);
  cout << "$Sequential_time " << timer.stop() << endl;

  OCLMatMulTask ocl_task;
  ocl_task.RunOCLMatMulForKernel(
    A_mat, wA, hA,
    B_mat, wB, hB,
    C_mat, wC, hC,
    wB, "gpu", wA, hA);

  free(A_mat);
  free(B_mat);
  free(C_mat);

}
