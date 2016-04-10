#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <math.h>
#include "boilerplate.cpp"
#include "../../../utils/helper.cpp"

using namespace std;
double gradient2(
		double i_level_grad,
		double i_index_grad,
		double j_level_grad,
		double j_index_grad,
		double lcl_q_inv
		) {
  double grad;
  
  //only affects the diagonal of the stiffness matrix
  unsigned doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
  grad = i_level_grad * 2.0 * (double)(doGrad);

  return (grad)* lcl_q_inv;
}

double gradient(
		double i_level_grad,
		double i_index_grad,
		double j_level_grad,
		double j_index_grad,
		double lcl_q_inv
		) {
  double grad;
  
  grad = i_level_grad * 2.0;

  return (grad)* lcl_q_inv;
}

double l2dot2(double lid,
	     double ljd,
	     double iid,
	     double ijd,
	     double in_lid,
	     double in_ljd,
	     double lcl_q
	     ) {

  double res_one = (2.0 / 3.0) * in_lid * (iid == ijd);

  bool selector = (lid > ljd);
  double i1d = iid * (selector) + ijd * (!selector);
  double in_l1d = in_lid * (selector) + in_ljd * (!selector);
  double i2d = ijd * (selector) + iid * (!selector);
  double l2d = ljd * (selector) + lid * (!selector);
  double in_l2d = in_ljd * (selector) + in_lid * (!selector);

  double q = (i1d - 1) * in_l1d;
  double p = (i1d + 1) * in_l1d;
  unsigned overlap = (std::max(q, (i2d - 1) * in_l2d) < std::min(p, (i2d + 1) * in_l2d));


  double temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  double res_two = temp_res * overlap; // Now mask result

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}

double l2dot(double lid,
	     double ljd,
	     double iid,
	     double ijd,
	     double in_lid,
	     double in_ljd,
	     double lcl_q
	     ) {

  double res_one = (2.0 / 3.0) * in_lid ;

  bool selector = (lid > ljd);
  double i1d = iid * (selector) + ijd * (!selector);
  double in_l1d = in_lid * (selector) + in_ljd * (!selector);
  double i2d = ijd * (selector) + iid * (!selector);
  double l2d = ljd * (selector) + lid * (!selector);
  double in_l2d = in_ljd * (selector) + in_lid * (!selector);

  double q = (i1d - 1) * in_l1d;
  double p = (i1d + 1) * in_l1d;
  unsigned overlap = (std::max(q, (i2d - 1) * in_l2d) < std::min(p, (i2d + 1) * in_l2d));


  double temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
  temp_res *= (0.5 * in_l1d);
  double res_two = temp_res * overlap; // Now mask result

  return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q;
}

void
Laplace(double * level,
	double * level_int,
	double * index,
	double * result,
	double * lcl_q_inv,
	double * lcl_q,
	double * alpha,
	double * lambda,
	unsigned storagesize, unsigned dim)
{

  for (unsigned i = 0; i < storagesize; i++) {
    double sub = 0.0;
    for (unsigned j = 0; j < storagesize; j++) {
      double gradient_temp[dim];
      double dot_temp[dim];
      for (unsigned d = 0; d < dim; d++) {
	double level_i = level[i * dim + d];
	double level_j = level[j * dim + d];
	double level_int_i = level_int[i * dim + d];
	double level_int_j = level_int[j * dim + d];
	double index_i = index[i * dim + d];
	double index_j = index[j * dim + d];
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
	double element = alpha[j];


	for (size_t d_inner = 0; d_inner < dim; d_inner++) {
	  element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
	}
//cout << "element " << element << endl;
//cout << "lambda[d_outer] " << lambda[d_outer] << endl;
	sub += lambda[d_outer] * element;
      }


    }
//    cout << "sub " << sub << endl;
    result[i] = sub;
  }
}

void
zeroMatrix(double* B, unsigned wB, unsigned hB)
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
randMat(double* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = ((rand() % ri) / rd ) * rd;
  }
}
void
divMat(double* mat, unsigned mat_size, double diver)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    mat[i] = mat[i] / diver;
  }
}

void
printMat(double* mat, unsigned mat_size)
{
  for (unsigned i = 0; i < mat_size; ++i) {
    cout << mat[i] << " ";
    if (i % 10 == 0) {
      cout << endl;
    }
  }
  cout << endl;
}


// #define storage_size 166400
// #define storage_size 16640
// #define dim 3

int main(int argc, char** argv)
{

  unsigned storage_size;
  unsigned dim;
  ParseCommandLine(argc, argv, &storage_size, NULL, &dim, NULL);
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

  
  double * level = new double[level_size];
  double * level_int = new double[level_int_size];
  double * index = new double[index_size];
  double * result_cpu = new double[result_size];
  double * result = new double[result_size];
  double * alpha = new double[alpha_size];
  double * lcl_q = new double[dim];
  double * lcl_q_inv = new double[dim];
  double * lambda = new double[dim];
  srand(2013);
  randMat(level, level_size);
  randMat(level_int, level_int_size);
  randMat(index, index_size); 
  randMat(lcl_q, dim);
  randMat(lcl_q_inv, dim);
  randMat(lambda, dim);
  randMat(alpha, alpha_size); 
  zeroMatrix(result_cpu, 1, result_size);
  zeroMatrix(result, 1, result_size);
  divMat(alpha, alpha_size, 100.0);
  divMat(level_int, level_int_size, 10000.0);


  Stopwatch timer;
  timer.start();
  Laplace(level,
  	  level_int,
  	  index,
  	  result_cpu,
  	  lcl_q_inv,
  	  lcl_q,
  	  alpha,
  	  lambda,
  	  storage_size,  dim);
  cout << "$Time " << timer.stop() << endl;  

  OCLLaplaceTask ocl_task;
  ocl_task.RunOCLLaplaceForKernel(alpha, alpha_size, dim,
	index, wIndex, hIndex,
	lambda, dim, lcl_q, dim, lcl_q_inv, dim,
	level, wLevel, hLevel,
	level_int, wLevel_int, hLevel_int,
	"gpu", result, result_size,
	storage_size);


  // printMat(alpha, alpha_size);
    
  helper::check_matrices(result_cpu, result, result_size);
//  printMat(result_cpu, 100);
//#endif
  // printMat(result, result_size);

  free(level);
  free(level_int);
  free(index);
  
}
