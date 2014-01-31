#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <openacc.h>


using namespace std;
#include <sys/time.h>
#define gettime(a) gettimeofday(a,NULL)
#define usec(t1,t2) (((t2).tv_sec-(t1).tv_sec)*1000000+((t2).tv_usec-(t1).tv_usec))
typedef struct timeval timestruct;

void
Jacobi(float*  B, float*  X1, float*  X2, unsigned wA, unsigned wB)
{
  #if 1
#pragma acc kernels loop copyin(B[0:wB*wB], X1[0:wA*wA]) local(X2[0:wA*wA]) independent
  #endif
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
copyMat(float* mat1, float* mat2, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < hMat; i++) {
    for (unsigned j = 0; j < wMat; j++) {
      mat1[i * wMat + j] = mat2[i * wMat + j];
    }
  }
}


#define matsize 8192

int main(int argc, char** argv)
{
  timestruct t1, t2;
  long long cgpu;

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
  acc_init( acc_device_nvidia );

  // printMat2(B_mat, wB, hB);
// 376082
#if 1
  gettime( &t1 );
  Jacobi(B_mat, X1_mat, X2_mat, wA, wB);
  gettime( &t2 );
  cgpu = usec(t1,t2);

#else 
#endif
  for (unsigned i = 0; i < 10; i++) {
    for (unsigned j = 0; j < 10; j++) {
      cout << X2_mat[i * wA + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
  // printMat(X2_mat, 100);
  printf( "matrix %d x %d, %d iterations\n", matsize, matsize, 1);
  printf( "%f seconds\n", (double)cgpu / 1000000.0);

  free(X1_mat);
  free(B_mat);
  free(X2_mat);
  
}