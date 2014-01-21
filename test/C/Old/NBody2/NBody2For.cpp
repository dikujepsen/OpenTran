for (unsigned i = 0; i < N; i++) {
  for (unsigned j = 0; j < N; j++) {
    float b_x = Pos[0][j];
    float b_y = Pos[1][j];
    float b_m = Mas[j];
    float a_x = Pos[0][i];
    float a_y = Pos[1][i];
    float a_m = Mas[i];

    float r_x = b_x - a_x;
    float r_y = b_y - a_y;
    float d = r_x * r_x + r_y * r_y;
    float deno = sqrt(d * d * d) + (i == j);
    deno = (a_m*b_m/deno) * (i != j);
    Forces_x[i*N + j] = deno * r_x;
    Forces_y[i*N + j] = deno * r_y;
  }
 }
