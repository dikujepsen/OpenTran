void JacobiFor(
	float * X2, size_t hst_ptrX2_dim1, size_t hst_ptrX2_dim2, 
	float * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2, 
	float * X1, size_t hst_ptrX1_dim1, size_t hst_ptrX1_dim2, 
	unsigned wB, unsigned wA)
{
  for (unsigned i = 1; i < wB; i++)
    {
      for (unsigned j = 1; j < wB; j++)
        {
          X2[i][j] = (-0.25) * ((B[i][j] - (X1[i - 1][j] + X1[i + 1][j])) - (X1[i][j - 1] + X1[i][j + 1]));
        }
    }
}

