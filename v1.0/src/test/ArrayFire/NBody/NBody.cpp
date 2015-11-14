#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <arrayfire.h>
#include <af/utils.h>

using namespace std;
using namespace af;


void
createMasses(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % 100) / 100.0) * 500.0;
    }
  }

}

void
createPosses(float* M, unsigned wM, unsigned hM)
{
  for (unsigned i = 0; i < hM; i++) {
    for (unsigned j = 0; j < wM; j++) {
      M[i * wM + j] = ((rand() % 100000) / 100000.0) * 100000.0;
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

  for (unsigned i = 0; i < N; i++) {
    float a_x = Pos[i]; 
    float a_y = Pos[N + i];
    float a_m = Mas[i];
    float f_x = 0;
    float f_y = 0;
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
    // printf("f_x[%d,%d] = %f\n" , i ,0, f_x);
    Forces[i] = f_x;
    Forces[N + i] = f_y;
  }

}


#define matsize 8192

int main(int argc, char** argv)
{
  int device = argc > 1 ? atoi(argv[1]) : 0;
  deviceset(device);
  info();
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
  


#if 0
  for (unsigned i = 0; i < 1; i++) {
    computeForces(Forces, Pos, M_mat, N);

  }
#else

  array F(wVel, hVel, Forces);
  array P(wPos, hPos, Pos);
  array M(hM, wM, M_mat);
  printf("\nBenchmarking........\n\n");
  af::sync();
  timer::start();
  // gfor (array k, N) {
  //   float a_x = P(0, 0).scalar<float>();
  //   float a_y = P(1, 0).scalar<float>();
  //   float a_m = M(0).scalar<float>();
  //   float f_x = 0.0;
  //   float f_y = 0.0;
    

  //   for (unsigned j = 0; j < N; j++) {
  //     float b_x = P(0, j).scalar<float>();
  //     float b_y = P(1, j).scalar<float>();
  //     float b_m = M(j).scalar<float>();
	
  //     float r_x = b_x - a_x;
  //     float r_y = b_y - a_y;

  //     float d = r_x * r_x + r_y * r_y;
  //     array ja = (constant(j,1));
  //     array mask1 = (k == ja);
  //     float deno = sqrt(d * d * d) + mask1.as(f32).scalar<float>();
  //     deno = (a_m*b_m/deno) * (!mask1).as(f32).scalar<float>();
  //     f_x += (deno * r_x) ;
  //     f_y += (deno * r_y) ;
  //   }
    
  //   F(0, k) = f_x;
  //   F(1, k) = f_y;
    
  // }

  gfor (array k, N) {
    array a_x = P(k, 0);
    array a_y = P(k, 1);
    array a_m = M(k);
    array f_x = constant(0,1);
    array f_y = constant(0,1);
    

    for (unsigned j = 0; j < N; j++) {
      array b_x = P(j, 0);
      array b_y = P(j, 1);
      array b_m = M(j);
      
      array r_x = b_x - a_x;
      array r_y = b_y - a_y;
      array d = r_x * r_x + r_y * r_y;
      array ja = (constant(j,1));
      array mask1 = (k == ja);
      array deno = sqrt(d * d * d) + mask1.as(f32);
      deno = (a_m*b_m/deno) * (!mask1).as(f32);
      // array dr = deno * r_x;
      f_x += (deno * r_x);
      f_y += (deno * r_y) ;
    }
    F(k, 0) = f_x;
    F(k, 1) = f_y;
    
  }
  // print(F);
  af::sync();
  printf("Average time : %g seconds\n", timer::stop());
  
  // Forces = F.host<float>();
#endif  


  // printMat2(Forces, wVel, hVel);

  free(Pos);
  free(M_mat);
  free(Vel);
  free(Forces);
  
}

// 1620.000000 670.000000 4660.000000 13590.000000 27460.000000 46270.000000 70020.000000 98710.000000 0.000000 0.000000
