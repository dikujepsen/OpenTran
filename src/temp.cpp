void matmulfunc4 (
	float * A, size_t hst_ptrA_dim1, size_t hst_ptrA_dim2, 
	float * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2, 
	float * C, size_t hst_ptrC_dim1, size_t hst_ptrC_dim2, 
	size_t hA, size_t wB, size_t wA)
{
  for (size_t i = 0; i < hA; i++)
    {
      for (size_t j = 0; j < wB; j++)
        {
          float sum = 0;
          for (size_t k = 0; k < wA; k++)
            {
              sum += A[i][k] * B[k][j];
            }
          C[i][j] = sum;
        }
    }
}

