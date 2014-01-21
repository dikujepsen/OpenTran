std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void NBody2For(" << endl;
  str << "	__global float * Mas, __global float * Pos, __global float * Forces_y, " << endl;
  str << "	__global float * Forces_x) {" << endl;
  str << "  float b_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float b_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float b_m = Mas[get_global_id(0)];" << endl;
  str << "  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(1)];" << endl;
  str << "  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(1)];" << endl;
  str << "  float a_m = Mas[get_global_id(1)];" << endl;
  str << "  float r_x = b_x - a_x;" << endl;
  str << "  float r_y = b_y - a_y;" << endl;
  str << "  float d = (r_x * r_x) + (r_y * r_y);" << endl;
  str << "  float deno = (sqrt((d * d) * d)) + (get_global_id(1) == get_global_id(0));" << endl;
  str << "  deno = ((a_m * b_m) / deno) * (get_global_id(1) != get_global_id(0));" << endl;
  str << "  Forces_x[(get_global_id(1) * hst_ptrForces_x_dim1) + get_global_id(0)] = deno * r_x;" << endl;
  str << "  Forces_y[(get_global_id(1) * hst_ptrForces_y_dim1) + get_global_id(0)] = deno * r_y;" << endl;
  str << "}" << endl;
  
  return str.str();
}

