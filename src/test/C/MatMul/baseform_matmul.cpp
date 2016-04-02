void MatMulFor(
  float * A, size_t hst_ptrA_dim1, size_t hst_ptrA_dim2,
  float * C, size_t hst_ptrC_dim1, size_t hst_ptrC_dim2,
  float * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2,
  unsigned hA, unsigned wB, unsigned wA
)
{
  for (unsigned i = 0; i < hA; i++)
  {
    for (unsigned j = 0; j < wB; j++)
    {
      float sum = 0;
      for (unsigned k = 0; k < wA; k++)
      {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

