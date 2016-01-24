#include "LaplaceIncludes.hpp"
double * index;
double * level_int;
double * level;
double * lcl_q;
double * result;
double * lcl_q_inv;
double * alpha;
double * lambda;
size_t dim;
size_t storagesize;
for (unsigned i = 0; i < storagesize; i++) {
  for (unsigned j = 0; j < storagesize; j++) {
    double gradient_temp[dim];
    double dot_temp[dim];
    for (unsigned d = 0; d < dim; d++) {
      double level_i = level[i][d];
      double level_j = level[j][d];
      double level_int_i = level_int[i][d];
      double level_int_j = level_int[j][d];
      double index_i = index[i][d];
      double index_j = index[j][d];
      gradient_temp[d] = gradient(level_i,index_i,
				   level_j,index_j, lcl_q_inv[d]);
      dot_temp[d] = l2dot(level_i,
			   level_j,
			   index_i,
			   index_j,
			   level_int_i,
			   level_int_j,
			   lcl_q[d]);
    }
    double sub = 0.0;
    for (size_t d_outer = 0; d_outer < dim; d_outer++) {
      double element = alpha[j];

      for (size_t d_inner = 0; d_inner < dim; d_inner++) {
	element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
      }
      sub += lambda[d_outer] * element;
    }
    result[i] += sub;
  }
 }
