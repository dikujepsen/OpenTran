void JacobiFor(
	float * X2, size_t hst_ptrX2_dim1, size_t hst_ptrX2_dim2, 
	float * B, size_t hst_ptrB_dim1, size_t hst_ptrB_dim2, 
	float * X1, size_t hst_ptrX1_dim1, size_t hst_ptrX1_dim2, 
	unsigned wB, unsigned wA)
{
  for (unsigned i = 1; i < wB; i++)
    {
      for (unsigned j = 1; j < wA; j++)
        {
          __local float X1_local[6 * 6];
          unsigned li = (get_local_id(1)) + 1;
          unsigned lj = (get_local_id(0)) + 1;
          
          X1_local[li - 1][lj] = X1[i - 1][j];
          X1_local[li + 1][lj] = X1[i + 1][j];
          X1_local[li][lj - 1] = X1[i][j - 1];
          X1_local[li][lj + 1] = X1[i][j + 1];
          barrier(CLK_LOCAL_MEM_FENCE);
          
          X2[i][j] = (-0.25) * ((B[i][j] - (X1[li - 1][lj] + X1[li + 1][lj])) - (X1[li][lj - 1] + X1[li][lj + 1]));
        }
    }
}

