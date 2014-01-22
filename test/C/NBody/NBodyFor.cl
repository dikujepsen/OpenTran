__kernel void NBodyFor(
	__global float * Mas, __global float * Pos, __global float * Forces
	) {
  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];
  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];
  float a_m = Mas[get_global_id(0)];
  float f_x = 0;
  float f_y = 0;
  for (unsigned j = 0; j < N; j++) {
      float b_x = Pos[(0 * hst_ptrPos_dim1) + j];
      float b_y = Pos[(1 * hst_ptrPos_dim1) + j];
      float b_m = Mas[j];
      float r_x = b_x - a_x;
      float r_y = b_y - a_y;
      float d = (r_x * r_x) + (r_y * r_y);
      float deno = (sqrt((d * d) * d)) + (get_global_id(0) == j);
      deno = ((a_m * b_m) / deno) * (get_global_id(0) != j);
      f_x += deno * r_x;
      f_y += deno * r_y;
  }
  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;
  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;
}
