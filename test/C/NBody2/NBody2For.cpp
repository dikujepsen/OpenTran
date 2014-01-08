for (unsigned i = 0; i < N; i++) {
  float f_x = 0;
  float f_y = 0;
  for (unsigned j = 0; j < N; j++) {
    float b_m = Mas[i] * Mas[j];

    float r_x = Pos[0][j] - Pos[0][i];
    float r_y = Pos[1][j] - Pos[1][i];
    float d = r_x * r_x + r_y * r_y;
    float deno = sqrt(d * d * d) + (i == j);
    deno = (b_m/deno) * (i != j);
    f_x += deno * r_x ;
    f_y += deno * r_y ;
  }

  Forces[0][i] = f_x;
  Forces[1][i] = f_y;
 }
