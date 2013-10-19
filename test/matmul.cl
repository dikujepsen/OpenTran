__kernel void
matmul(__global float* dev_A, __global float* dev_B, __global float* dev_C,
       unsigned hA, unsigned wA, unsigned wB)
{
  float sum = 0;
  for (unsigned k = 0; k < wA; ++k) {
    sum += dev_A[get_global_id(1) * wA + k] * dev_B[k * wB + get_global_id(0)];
  }
  dev_C[get_global_id(1) * wB + get_global_id(0)] = sum;
}
