std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void NBodyFor(" << endl;
  str << "	__global float * Mas, __global float * Pos, __global float * Forces" << endl;
  str << "	) {" << endl;
  str << "  __local float Mas_local[256];" << endl;
  str << "  __local float Pos_local[256];" << endl;
  str << "  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float a_m = Mas[get_global_id(0)];" << endl;
  str << "  float f_x = 0;" << endl;
  str << "  float f_y = 0;" << endl;
  str << "  for (unsigned j = 0; j < N; j+=256) {" << endl;
  str << "      Mas_local[get_local_id(1)] = Mas[j + get_local_id(1)];" << endl;
  str << "      Pos_local[(0 * 256) + get_local_id(0)] = Pos[(0 * hst_ptrPos_dim1) + j + get_local_id(0)];" << endl;
  str << "      Pos_local[(1 * 256) + get_local_id(0)] = Pos[(1 * hst_ptrPos_dim1) + j + get_local_id(0)];" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "      for (unsigned jj = 0; jj < 256; jj++) {" << endl;
  str << "          float b_x = Pos_local[(0 * 256) + jj];" << endl;
  str << "          float b_y = Pos_local[(1 * 256) + jj];" << endl;
  str << "          float b_m = Mas_local[jj];" << endl;
  str << "          float r_x = b_x - a_x;" << endl;
  str << "          float r_y = b_y - a_y;" << endl;
  str << "          float d = (r_x * r_x) + (r_y * r_y);" << endl;
  str << "          float deno = (sqrt((d * d) * d)) + (get_global_id(0) == j);" << endl;
  str << "          deno = ((a_m * b_m) / deno) * (get_global_id(0) != j);" << endl;
  str << "          f_x += deno * r_x;" << endl;
  str << "          f_y += deno * r_y;" << endl;
  str << "      }" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "  }" << endl;
  str << "  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;" << endl;
  str << "  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;" << endl;
  str << "}" << endl;
  
  return str.str();
}

