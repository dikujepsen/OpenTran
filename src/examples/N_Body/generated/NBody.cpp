#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "boilerplate.cpp"
#include "helper.cpp"


using namespace std;


void
createMasses(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % 100) / 10.0) * 500.0;
    }
  }

}

#define ri 9000
#define rd 9000.0
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




int main(int argc, char** argv)
{
  unsigned matsize = 8192;

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
  float* Forces_cpu = new float[Vel_size];
  float* Forces = new float[Vel_size];

  srand(2553);
  createMasses(M_mat, wM, hM);
  createPosses(Pos, wPos, hPos);
  createVelles(Vel, wVel, hVel);


  Stopwatch timer;
  timer.start();
  computeForces(Forces_cpu, Pos, M_mat, N);
  cout << "$Sequential_time " << timer.stop() << endl;  


  OCLNBodyTask ocl_task;
  ocl_task.RunOCLNBodyForKernel(Forces, wVel, 2,
	M_mat, wM, N,
	Pos, wPos, 2,
	"gpu");


  free(Pos);
  free(M_mat);
  free(Vel);
  free(Forces);
  
}
