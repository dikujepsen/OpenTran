#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "boilerplate.cpp"


using namespace std;


void
createMasses(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % 100) / 100.0) * 500.0;
    }
  }

}

#define ri 100000
#define rd 100000.0
void
createPosses(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % ri) / rd ) * rd;
    }
  }

}

void
createVelles(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % 100) / 100.0);
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
printMat3(float* mat, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < (hMat); i++) {
    cout << mat[i * wMat] << " ";
  }
  cout << endl;
}

void
printMat4(float* mat, unsigned wMat, unsigned hMat, unsigned jnum)
{
  for (unsigned i = 0; i < (hMat); i++) {
    for (unsigned j = 0; j < (jnum); j++) {
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


void
summation(float* mat1, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < hMat; i++) {
    for (unsigned j = 1; j < wMat; j++) {
      mat1[i * wMat] += mat1[i * wMat + j];
    }
  }
}
void
summation2(float* mat1, unsigned wMat, unsigned hMat, unsigned jnum)
{
  for (unsigned i = 0; i < hMat; i++) {
    for (unsigned j = 1; j < jnum; j++) {
      mat1[i * wMat] += mat1[i * wMat + j];
    }
  }
}

void computeForces(float * Forces, float * Pos, float * Mas, unsigned N);

void VelStorVer(float * Forces, float * Pos, float * Vel,
		float * Mas, float dt, unsigned N) {
  for (unsigned i = 0; i < N; i++) {
    float m = Mas[i];
    float a_x = Forces[i]/m;
    float a_y = Forces[N + i]/m;

    float v_x_new = Vel[i] + 0.5*dt*a_x;
    float v_y_new = Vel[N + i] + 0.5*dt*a_y;

    Pos[i]     += dt*v_x_new;
    Pos[N + i] += dt*v_y_new;

    Vel[i] = v_x_new;
    Vel[N + i] = v_y_new;
  }

  // computeForces(Forces, Pos, Mas, N);

  for (unsigned i = 0; i < N; i++) {
    float a_x = Forces[i]/Mas[i];
    float a_y = Forces[N + i]/Mas[i];
    Vel[i] += 0.5*dt*a_x;
    Vel[N + i] += 0.5*dt*a_y;
  }
    
}

void computeForces(float * Forces_x, float * Forces_y, float * Pos, float * Mas, unsigned N) {

  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
      float b_x = Pos[j]; 
      float b_y = Pos[N + j];
      float b_m = Mas[j];
      float a_x = Pos[i]; 
      float a_y = Pos[N + i];
      float a_m = Mas[i];

      float r_x = b_x - a_x;
      float r_y = b_y - a_y;
      float d = r_x * r_x + r_y * r_y;
      float deno = sqrt(d * d * d) + (i == j);
      deno = (a_m*b_m/deno) * (i != j);
      Forces_x[i*N + j] = deno * r_x;
      Forces_y[i*N + j] = deno * r_y;
    }
  }
  
}


#define matsize 20480
// #define matsize 32

int main(int argc, char** argv)
{
  unsigned hPos = 2;
  unsigned hM = 1;
  unsigned hVel = 2;
  unsigned wPos = matsize;
  unsigned wM = matsize;
  unsigned wVel = matsize;

  unsigned N = matsize;
  
  unsigned Pos_size = hPos*wPos;
  unsigned M_size = hM*wM;
  unsigned Vel_size = hVel*wVel;
  unsigned Forces_size = N*N;
  
  float* M_mat = new float[M_size];
  float* Pos = new float[Pos_size];
  float* Vel = new float[Vel_size];
  float* Forces_x = new float[Forces_size];
  float* Forces_y = new float[Forces_size];

  srand(2553);
  createMasses(M_mat, wM, hM);
  createPosses(Pos, wPos, hPos);
  createVelles(Vel, wVel, hVel);


#define GPU 1
#if GPU
  
RunOCLNBody2ForKernel(
	M_mat, wM, Forces_y, 
	N, N, Pos, 
	wPos, 2, Forces_x, 
	N, N, N);

  
#else
  computeForces(Forces_x, Forces_y, Pos, M_mat, N);

#endif

  // printMat2(M_mat, wM, hM);
  // printMat2(Pos   , wPos, hPos);
  // printMat2(Vel   , wVel, hVel);

  // summation(Forces_x, N, N);
  // summation(Forces_y, N, N);
  // printMat3(Forces_x, N, N);
  // printMat3(Forces_y, N, N);
  
  // printMat2(Forces_x, N, N);
  // printMat2(Forces_y, N, N);
  // summation2(Forces_x, N, N, N/8);
  // summation2(Forces_y, N, N, N/8);
  // printMat3(Forces_x, N, N);
  // printMat3(Forces_y, N, N);

  free(Pos);
  free(M_mat);
  free(Vel);
  free(Forces_x);
  free(Forces_y);
  
}
