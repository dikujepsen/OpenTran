#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "../../../utils/Stopwatch.cpp"
Stopwatch timer;
#include <openacc.h>


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
copyMat(float* mat1, float* mat2, unsigned wMat, unsigned hMat)
{
  for (unsigned i = 0; i < hMat; i++) {
    for (unsigned j = 0; j < wMat; j++) {
      mat1[i * wMat + j] = mat2[i * wMat + j];
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

  computeForces(Forces, Pos, Mas, N);

  for (unsigned i = 0; i < N; i++) {
    float a_x = Forces[i]/Mas[i];
    float a_y = Forces[N + i]/Mas[i];
    Vel[i] += 0.5*dt*a_x;
    Vel[N + i] += 0.5*dt*a_y;
  }
    
}

void computeForces(float * Forces, float * Pos, float * Mas, unsigned N) {

#ifndef CPU
#ifdef READBACK
#pragma acc kernels loop copyin(Pos[0:2*N], Mas[0:N])  copyout(Forces[0:N] ) independent
#else
#pragma acc kernels loop copyin(Pos[0:2*N], Mas[0:N])  local(Forces[0:N] ) independent
#endif
#endif
  for (unsigned i = 0; i < N; i++) {
    float a_x = Pos[i]; 
    float a_y = Pos[N + i];
    float a_m = Mas[i];
    float f_x = 0.0;
    float f_y = 0.0;
    for (unsigned j = 0; j < N; j++) {
      float b_x = Pos[j]; 
      float b_y = Pos[N + j];
      float b_m = Mas[j];

      float r_x = b_x - a_x;
      float r_y = b_y - a_y;
      float d = r_x * r_x + r_y * r_y;
      float deno = sqrt(d * d * d) + (i == j);
      deno = (a_m*b_m/deno) * (i != j);
      f_x += deno * r_x ;
      f_y += deno * r_y ;
    }
    Forces[i] = f_x;
    Forces[N + i] = f_y;
  }

}


// #define matsize 256000
// #define matsize 76800
//#define matsize 256

int main(int argc, char** argv)
{
  unsigned matsize;
  ParseCommandLine(argc, argv, &matsize, NULL, NULL);
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
  
  float* M_mat = new float[M_size];
  float* Pos = new float[Pos_size];
  float* Vel = new float[Vel_size];
  float* Forces = new float[Vel_size];

  srand(2553);
  createMasses(M_mat, wM, hM);
  createPosses(Pos, wPos, hPos);
  createVelles(Vel, wVel, hVel);

  // computeForces(Forces, Pos, M_mat, N);
  // float dt = 0.015;
  acc_init( acc_device_nvidia );


  timer.start();
  computeForces(Forces, Pos, M_mat, N);
  cout << timer.stop() << endl;
 
#if PRINT
  printMat(Forces, 100);
#endif

  free(Pos);
  free(M_mat);
  free(Vel);
  free(Forces);
  
}

