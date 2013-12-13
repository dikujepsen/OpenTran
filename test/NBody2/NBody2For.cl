#define LSIZE 4

__kernel void NBody2For(
	unsigned hst_ptrForces_dim1, __global float * Mas, __global float * Pos, 
	unsigned N, __global float * Forces, unsigned hst_ptrPos_dim1)
{
  __local float Pos_local[2*4];
  unsigned li = get_local_id(0);
  
  float f_x = 0;
  float f_y = 0;
  for (unsigned j = 0; j < N; j+=4)
    {
      Pos_local[(0 * 4) + li] = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];
      Pos_local[(1 * 4) + li] = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];
       barrier(CLK_LOCAL_MEM_FENCE);
      
      for (unsigned jj = 0; jj < 4; jj++)
        {
          float b_m = Mas[get_global_id(0)] * Mas[j + jj];
          float a_x = Pos_local[(0 * 4) + li];
          float a_y = Pos_local[(1 * 4) + li];
          float r_x = Pos_local[(0 * 4) + (j + jj)] - a_x;
          float r_y = Pos_local[(1 * 4) + (j + jj)] - a_y;
          float d = (r_x * r_x) + (r_y * r_y);
          float deno = (sqrt((d * d) * d)) + (get_global_id(0) == (j + jj));
          deno = (b_m / deno) * (get_global_id(0) != (j + jj));
          f_x += deno * r_x;
          f_y += deno * r_y;
        }
    }
  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;
  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;
}
