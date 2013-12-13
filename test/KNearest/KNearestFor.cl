#define LSIZE 8

__kernel void KNearestFor(			  unsigned dim, unsigned hst_ptrtest_patterns_dim1, __global float * dist_matrix, 
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
      train_patterns_local[(li * 8) + get_local_id(0)] = train_patterns[((k + get_local_id(1)) * hst_ptrtrain_patterns_dim1) + get_global_id(0)];
      barrier(CLK_LOCAL_MEM_FENCE);
      
      /* for (unsigned kk = 0; kk < 8; kk++) */
      /*   { */
          float tmp = test_patterns_local[(li * 8) + 0] - train_patterns_local[(0 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 1] - train_patterns_local[(1 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 2] - train_patterns_local[(2 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 3] - train_patterns_local[(3 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 4] - train_patterns_local[(4 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 5] - train_patterns_local[(5 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 6] - train_patterns_local[(6 * 8) + lj];
          d += tmp * tmp;
          tmp = test_patterns_local[(li * 8) + 7] - train_patterns_local[(7 * 8) + lj];
          d += tmp * tmp;
        /* } */
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  
  dist_matrix[(get_global_id(1) * hst_ptrdist_matrix_dim1) + get_global_id(0)] = d;
}
