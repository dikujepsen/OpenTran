#include <iostream>
#include <arrayfire.h>
#include <stdio.h>
#include <assert.h>

using namespace af;





#define MB (1024 * 1024)
#define mb(x)   (unsigned)((x) / MB + !!((x) % MB))

int main(int argc, char  **argv)
{
  try {
    int n = 16;
    printf("size(A)=[%d,%d]  (%u mb)\n", n, n, mb(n * n * sizeof(float)));

        
    array A = constant(1,n,n);
    array B = constant(1,n);
    array C = constant(0,n);
    array C2 = constant(0,n);
    array B2 = randn(n,1);

    float *Y = new float[n];
	
	
    printf("\nBenchmarking........\n\n");
    af::sync();
    timer::start();
    gfor (array k, n) {
      C(k) = sum(A(span, k) * B2);  // matrix-vector multiply
    }
    af::sync();
    printf("Average time : %g seconds\n", timer::stop());

    Y = C.host<float>();
    for (int i = 0; i < n; i++) {
      printf("Y[%d] = %f\n", i, Y[i]);
    }
    printf("\n\n");
    C2 = matmul(A, B2);
    Y = C2.host<float>();
    for (int i = 0; i < n; i++) {
      printf("Y[%d] = %f\n", i, Y[i]);
    }

    // delete[] A; delete[] B; delete[] C; // cleanup


  } catch (af::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw;
  }

  return 0;
}
