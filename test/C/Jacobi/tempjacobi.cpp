void JacobiFor(
	unknown * X2, size_t hst_ptrX2_dim1, size_t hst_ptrX2_dim2, 
	unknown * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2, 
	unknown * X1, size_t hst_ptrX1_dim1, size_t hst_ptrX1_dim2, 
	unknown wB, unknown wA)
{
  for (unsigned i = 1; i < wB; i++)
    {
      for (unsigned j = 1; j < wB; j++)
        {
          X2[i][j] = (-0.25) * ((B[i][j] - (X1[((i - 1) * wA) + j] + X1[((i + 1) * wA) + j])) - (X1[(i * wA) + (j - 1)] + X1[(i * wA) + (j + 1)]));
        }
    }
}

