#define LSIZE 8

__kernel void KNearestFor2(
	__global float * dist_matrix, __global float * train_patterns, __global float * test_patterns)
{
  float test_patterns_reg[dim];
  for (unsigned k = 0; k < dim; k++)
    {
      test_patterns_reg[k] = test_patterns[(k * hst_ptrtest_patterns_dim1) + get_global_id(0)];
    }
  
  float z = 0.0;
  for (unsigned j = 0; j < NTRAIN; j++)
    {
      float d = 0.0;
      for (unsigned k = 0; k < dim; k++)
        {
          float tmp = test_patterns_reg[k] - train_patterns[(j * hst_ptrtrain_patterns_dim1) + k];
          d += tmp * tmp;
        }
      dist_matrix[(j * hst_ptrdist_matrix_dim1) + get_global_id(0)] = d;
    }
}
