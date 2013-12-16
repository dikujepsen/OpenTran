#define LSIZE 8

__kernel void NBody2For(
	unsigned hst_ptrForces_dim1, __global float * Mas, __global float * Pos, 
	unsigned N, __global float * Forces, unsigned hst_ptrPos_dim1)
{
  float f_x = 0;
  float f_y = 0;
  for (unsigned j = 0; j < N; j+=8)
    {
      unsigned li = get_local_id(0);
      
      for (unsigned jj = 0; jj < 8; jj++)
        {
          float b_m = Mas[get_global_id(0)] * Mas[j];
          float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];
          float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];
          float r_x = Pos[(0 * hst_ptrPos_dim1) + j] - a_x;
          float r_y = Pos[(1 * hst_ptrPos_dim1) + j] - a_y;
          float d = (r_x * r_x) + (r_y * r_y);
          float deno = (sqrt((d * d) * d)) + (get_global_id(0) == j);
          deno = (b_m / deno) * (get_global_id(0) != j);
          f_x += deno * r_x;
          f_y += deno * r_y;
        }
    }
  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;
  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;
}
