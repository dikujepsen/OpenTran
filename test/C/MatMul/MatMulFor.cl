__kernel void MatMulFor(
	__global float * A, __global float * C, __global float * B
	) {
  __local float A_local[16*16];
  __local float B_local[16*16];
  float sum = 0;
  for (unsigned k = 0; k < wA; k+=16) {
      A_local[(get_local_id(1) * 16) + get_local_id(0)] = A[(get_global_id(1) * hst_ptrA_dim1) + k + get_local_id(0)];
      B_local[(get_local_id(1) * 16) + get_local_id(0)] = B[(k + get_local_id(1) * hst_ptrB_dim1) + get_global_id(0)];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (unsigned kk = 0; kk < 16; kk++) {
          sum += A_local[(get_local_id(1) * 16) + kk] * B_local[(kk * 16) + get_local_id(0)];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[(get_global_id(1) * hst_ptrC_dim1) + get_global_id(0)] = sum;
}
