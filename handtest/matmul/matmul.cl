__kernel void
matmul(__global float* dev_ptrA, __global float* dev_ptrB, __global float* dev_ptrC,
       unsigned hA, unsigned wA, unsigned wB)
{
  float sum = 0;
  for (unsigned k = 0; k < wA; ++k) {
    sum += dev_ptrA[get_global_id(1) * wA + k] * dev_ptrB[k * wB + get_global_id(0)];
  }
  dev_ptrC[get_global_id(1) * wB + get_global_id(0)] = sum;
}
