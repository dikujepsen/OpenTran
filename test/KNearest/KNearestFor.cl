#define LSIZE 8

__kernel void KNearestFor(
	unsigned dim, unsigned hst_ptrtest_patterns_dim1, __global float * dist_matrix, 
	__global float * train_patterns, unsigned hst_ptrtrain_patterns_dim1, __global float * test_patterns, 
	unsigned hst_ptrdist_matrix_dim1)
{
  __local float test_patterns_local[8*8];
  __local float train_patterns_local[8*8];
  unsigned li = get_local_id(1);
  unsigned lj = get_local_id(0);
  
  float d = 0.0;
  for (unsigned k = 0; k < dim; k+=8)
    {
      test_patterns_local[(li * 8) + get_local_id(0)] = test_patterns[(get_global_id(1) * hst_ptrtest_patterns_dim1) + (k + get_local_id(0))];
      train_patterns_local[(get_local_id(1) * 8) + lj] = train_patterns[((k + get_local_id(1)) * hst_ptrtrain_patterns_dim1) + get_global_id(0)];
       barrier(CLK_LOCAL_MEM_FENCE);
      
      for (unsigned kk = 0; kk < 8; kk++)
        {
          float tmp = test_patterns_local[(li * 8) + kk] - train_patterns_local[(kk * 8) + lj];
          d += tmp * tmp;
        }
    }
  dist_matrix[(get_global_id(1) * hst_ptrdist_matrix_dim1) + get_global_id(0)] = d;
}
