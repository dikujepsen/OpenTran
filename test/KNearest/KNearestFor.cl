#define LSIZE 4

__kernel void KNearestFor(
	unsigned dim, unsigned hst_ptrtest_patterns_dim1, __global float * dist_matrix, 
	__global float * train_patterns, unsigned hst_ptrtrain_patterns_dim1, __global float * test_patterns, 
	unsigned hst_ptrdist_matrix_dim1)
{
  float d = 0.0;
  for (size_t k = 0; k < dim; k++)
    {
      float tmp = test_patterns[(get_global_id(1) * hst_ptrtest_patterns_dim1) + k] - train_patterns[(get_global_id(0) * hst_ptrtrain_patterns_dim1) + k];
      d += tmp * tmp;
    }
  dist_matrix[(get_global_id(0) * hst_ptrdist_matrix_dim1) + get_global_id(1)] = d;
}
