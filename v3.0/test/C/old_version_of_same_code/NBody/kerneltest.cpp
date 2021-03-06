void NBodyFor(
	float * Mas, size_t hst_ptrMas_dim1, float * Pos, 
	size_t hst_ptrPos_dim1, size_t hst_ptrPos_dim2, float * Forces, 
	size_t hst_ptrForces_dim1, size_t hst_ptrForces_dim2, size_t N
	)
{
  for (unsigned i = 0; i < N; i++)
    {
      float f_x = 0;
      float f_y = 0;
      for (unsigned j = 0; j < N; j++)
        {
          float a_x = Pos[0][i];
          float a_y = Pos[1][i];
          float a_m = Mas[i];
          float b_x = Pos[0][j];
          float b_y = Pos[1][j];
          float b_m = Mas[j];
          float r_x = b_x - a_x;
          float r_y = b_y - a_y;
          float d = (r_x * r_x) + (r_y * r_y);
          float deno = (sqrt((d * d) * d)) + (i == j);
          deno = ((a_m * b_m) / deno) * (i != j);
          f_x += deno * r_x;
          f_y += deno * r_y;
        }
      Forces[0][i] = f_x;
      Forces[1][i] = f_y;
    }
}

