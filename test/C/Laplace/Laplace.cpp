#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <math.h>
#include "boilerplate.cpp"

using namespace std;
float gradient2(
		float i_level_grad,
		float i_index_grad,
		float j_level_grad,
		float j_index_grad,
		float lcl_q_inv
		) {
  float grad;
  
  //only affects the diagonal of the stiffness matrix
  unsigned doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
  grad = i_level_grad * 2.0 * (float)(doGrad);

  return (grad)* lcl_q_inv;
}

float gradient(
		float i_level_grad,
		float i_index_grad,
		float j_level_grad,
		float j_index_grad,
		float lcl_q_inv
		) {
  float grad;
  
  grad = i_level_grad * 2.0;

  return (grad)* lcl_q_inv;
}

float l2dot2(float lid,
	     float ljd,
	     float iid,
	     float ijd,
	     float in_lid,
	     float in_ljd,
	     float lcl_q
	     ) {

  float res_one = (2.0 / 3.0) * in_lid * (iid == ijd);

  bool selector = (lid > ljd);
  float i1d = iid * (selector) + ijd * (!selector);
  float in_l1d = in_lid * (selector) + in_ljd * (!selector);
  float i2d = ijd * (selector) + iid * (!selector);
  float l2d = ljd * (selector) + lid * (!selector);
  float in_l2d = in_ljd * (selector) + in_lid * (!selector);

  float q = (i1d - 1) * in_l1d;
  float p = (i1d + 1) * in_l1d;
  unsigned overlap = (std::max(q, (i2d - 1) * in_l2d) < std::min(p, (i2d + 1) * in_l2d));


  float temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  float res_two = temp_res * overlap; // Now mask result

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}

float l2dot(float lid,
	     float ljd,
	     float iid,
	     float ijd,
	     float in_lid,
	     float in_ljd,
	     float lcl_q
	     ) {

  float res_one = (2.0 / 3.0) * in_lid ;

  bool selector = (lid > ljd);
  float i1d = iid * (selector) + ijd * (!selector);
  float in_l1d = in_lid * (selector) + in_ljd * (!selector);
  float i2d = ijd * (selector) + iid * (!selector);
  float l2d = ljd * (selector) + lid * (!selector);
  float in_l2d = in_ljd * (selector) + in_lid * (!selector);

  float q = (i1d - 1) * in_l1d;
  float p = (i1d + 1) * in_l1d;
  unsigned overlap = (std::max(q, (i2d - 1) * in_l2d) < std::min(p, (i2d + 1) * in_l2d));


  float temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  float res_two = temp_res * overlap; // Now mask result

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}

void
Laplace(float * level,
	float * level_int,
	float * index,
	float * result,
	float * lcl_q_inv,
	float * lcl_q,
	float * alpha,
	float * lambda,
	unsigned storagesize, unsigned dim)
{

  for (unsigned i = 0; i < storagesize; i++) {
    float sub = 0.0;
    for (unsigned j = 0; j < storagesize; j++) {
      float gradient_temp[dim];
      float dot_temp[dim];
      for (unsigned d = 0; d < dim; d++) {
	float level_i = level[i * dim + d];
	float level_j = level[j * dim + d];
	float level_int_i = level_int[i * dim + d];
	float level_int_j = level_int[j * dim + d];
	float index_i = index[i * dim + d];
	float index_j = index[j * dim + d];
	gradient_temp[d] = gradient2(level_i,index_i,
				     level_j,index_j, lcl_q_inv[d]);
	dot_temp[d] = l2dot2(level_i,
			     level_j,
			     index_i,
			     index_j,
			     level_int_i,
			     level_int_j,
			     lcl_q[d]);
      }
      for (size_t d_outer = 0; d_outer < dim; d_outer++) {
	float element = alpha[j];

	for (size_t d_inner = 0; d_inner < dim; d_inner++) {
	  element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
	}
	sub += lambda[d_outer] * element;
      }
      
    }
    result[i] = sub;
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

#define ri 20
#define rd 20.0
void
randMat(float* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = ((rand() % ri) / rd ) * rd;
  }
}
void
divMat(float* mat, unsigned mat_size, float diver)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = mat[i] / diver;
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


#define storage_size 1024
#define dim 3

int main(int argc, char** argv)
{
  unsigned wLevel = dim;
  unsigned hLevel = storage_size;
  unsigned wLevel_int = dim;
  unsigned hLevel_int = storage_size;
  unsigned wIndex = dim;
  unsigned hIndex = storage_size;

  unsigned level_size = hLevel*wLevel;
  unsigned level_int_size = hLevel_int*wLevel_int;
  unsigned index_size = hIndex*wIndex;
  unsigned result_size = storage_size;
  unsigned alpha_size = storage_size;

  
  float * level = new float[level_size];
  float * level_int = new float[level_int_size];
  float * index = new float[index_size];
  float * result = new float[result_size];
  float * alpha = new float[alpha_size];
  float * lcl_q = new float[dim];
  float * lcl_q_inv = new float[dim];
  float * lambda = new float[dim];
  srand(2013);
  randMat(level, level_size);
  randMat(level_int, level_int_size);
  randMat(index, index_size); 
  randMat(lcl_q, dim);
  randMat(lcl_q_inv, dim);
  randMat(lambda, dim);
  randMat(alpha, alpha_size); 
  zeroMatrix(result, 1, result_size);
  divMat(alpha, alpha_size, 100.0);
  divMat(level_int, level_int_size, 10000.0);
 
#if 0
  Laplace(level,
  	  level_int,
  	  index,
  	  result,
  	  lcl_q_inv,
  	  lcl_q,
  	  alpha,
  	  lambda,
  	  storage_size,  dim);

#else
  RunOCLLaplaceForKernel(
			 dim, level_int, wLevel_int, 
			 hLevel_int, index, wIndex, hIndex, lcl_q, dim, 
			 level, wLevel, hLevel, 
			 result, result_size, lcl_q_inv, dim,
			 alpha, alpha_size, 
			 storage_size, lambda, dim
			 );
#endif
  // printMat(alpha, alpha_size);
    
  printMat(result, result_size);

  free(level);
  free(level_int);
  free(index);
  
}
