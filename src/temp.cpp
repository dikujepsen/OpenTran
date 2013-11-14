int matmulfunc4(
	unknown * A, size_t hst_ptrA_dim1, size_t hst_ptrA_dim2, 
	unknown * C, size_t hst_ptrC_dim1, size_t hst_ptrC_dim2, 
	unknown * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2, 
	unknown hA, unknown E, unknown wB, 
	unknown wA, unknown D)
{
  for (unsigned i = 0; i < hA; i++)
    {
      int b;
      int a = 4;
      for (unsigned j = 0; j < wB; j++)
        {
          float sum = 0;
          for (unsigned k = 0; k < wA; k++)
            {
              sum += (A[i][k] * B[k][j]) + D;
            }
          C[i][j] = sum * E;
        }
    }
}

