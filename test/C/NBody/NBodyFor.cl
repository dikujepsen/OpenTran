std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void NBodyFor(" << endl;
  str << "	__global float * Mas, __global float * Pos, __global float * Forces" << endl;
  str << "	) {" << endl;
  str << "  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float a_m = Mas[get_global_id(0)];" << endl;
  str << "  float f_x = 0;" << endl;
  str << "  float f_y = 0;" << endl;
  str << "  for (unsigned j = 0; j < N; j++) {" << endl;
  str << "      float b_x = Pos[(0 * hst_ptrPos_dim1) + j];" << endl;
  str << "      float b_y = Pos[(1 * hst_ptrPos_dim1) + j];" << endl;
  str << "      float b_m = Mas[j];" << endl;
  str << "      float r_x = b_x - a_x;" << endl;
  str << "      float r_y = b_y - a_y;" << endl;
  str << "      float d = (r_x * r_x) + (r_y * r_y);" << endl;
  str << "      float deno = (sqrt((d * d) * d)) + (get_global_id(0) == j);" << endl;
  str << "      deno = ((a_m * b_m) / deno) * (get_global_id(0) != j);" << endl;
  str << "      f_x += deno * r_x;" << endl;
  str << "      f_y += deno * r_y;" << endl;
  str << "  }" << endl;
  str << "  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;" << endl;
  str << "  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;" << endl;
  str << "}" << endl;
  
  return str.str();
}

