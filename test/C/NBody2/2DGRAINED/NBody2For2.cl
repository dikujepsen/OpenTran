#define LSIZE 256
__kernel void NBody2For(
			unsigned hst_ptrForces_x_dim1, __global float * Mas, __global float * Pos, 
			unsigned hst_ptrForces_y_dim1, __global float * Forces_y, __global float * Forces_x, 
			unsigned hst_ptrPos_dim1) {
  __local float Pos_x_local[LSIZE];
  __local float Pos_y_local[LSIZE];
  __local float Mas_local[LSIZE];
  Pos_x_local[get_local_id(0)] = Pos[(0 * hst_ptrPos_dim1) + LSIZE*get_group_id(1) + get_local_id(0)];
  Pos_y_local[get_local_id(0)] = Pos[(1 * hst_ptrPos_dim1) + LSIZE*get_group_id(1) + get_local_id(0)];
  Mas_local[get_local_id(0)] = Mas[LSIZE*get_group_id(1) + get_local_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];
  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];
  float a_m = Mas[get_global_id(0)];
  unsigned idx = LSIZE*get_group_id(1);
  float f_x = 0.0;
  float f_y = 0.0;
  for (unsigned k = 0; k < LSIZE; k++) {
    /* float b_x = Pos[(0 * hst_ptrPos_dim1) + LSIZE*get_group_id(1) + k]; */
    /* float b_y = Pos[(1 * hst_ptrPos_dim1) + LSIZE*get_group_id(1) + k]; */
    /* float b_m = Mas[LSIZE*get_group_id(1) + k]; */
    float b_x = Pos_x_local[k];
    float b_y = Pos_y_local[k];
    float b_m = Mas_local[k];
    float r_x = b_x - a_x;
    float r_y = b_y - a_y;
    float d = (r_x * r_x) + (r_y * r_y);
    float deno = (sqrt((d * d) * d)) + (get_global_id(0) == (idx+k));
    deno = ((a_m * b_m) / deno) * (get_global_id(0) != (idx+k));
    /* f_x = deno * r_x; */
    /* f_y = deno * r_y; */
    f_x += deno * r_x;
    f_y += deno * r_y;
  }
  /* float res_x = get_local_id(0) == 0 ? f_x : 0.0; */
  /* float res_y = get_local_id(0) == 0 ? f_y : 0.0; */
  /* float res_x = get_local_id(0) == 0 ? f_x : f_x; */
  Forces_x[(get_global_id(0) * hst_ptrForces_x_dim1) + get_group_id(1)] = f_x;
  Forces_y[(get_global_id(0) * hst_ptrForces_y_dim1) + get_group_id(1)] = f_y;
  /* float res_y = get_local_id(0) == 0 ? f_y : f_y; */
  /* float res_x = f_x; */
  /* float res_y = f_y; */
  /* Forces_x[(get_global_id(1) * hst_ptrForces_x_dim1) + get_global_id(0)] = res_x; */
  /* Forces_y[(get_global_id(1) * hst_ptrForces_y_dim1) + get_global_id(0)] = res_y; */
}
