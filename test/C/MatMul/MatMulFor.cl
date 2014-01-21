std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void MatMulFor(" << endl;
  str << "	__global float * A, __global float * C, __global float * B" << endl;
  str << "	) {" << endl;
  str << "  __local float A_local[16*16];" << endl;
  str << "  __local float B_local[16*16];" << endl;
  str << "  unsigned li = get_local_id(1);" << endl;
  str << "  unsigned lj = get_local_id(0);" << endl;
  str << "  float sum = 0;" << endl;
  str << "  for (unsigned k = 0; k < wA; k+=16) {" << endl;
  str << "      A_local[(li * 16) + get_local_id(0)] = A[(get_global_id(1) * hst_ptrA_dim1) + (k + get_local_id(0))];" << endl;
  str << "      B_local[(get_local_id(1) * 16) + lj] = B[((k + get_local_id(1)) * hst_ptrB_dim1) + get_global_id(0)];" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "      for (unsigned kk = 0; kk < 16; kk++) {" << endl;
  str << "          sum += A_local[(li * 16) + kk] * B_local[(kk * 16) + lj];" << endl;
  str << "      }" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "  }" << endl;
  str << "  C[(get_global_id(1) * hst_ptrC_dim1) + get_global_id(0)] = sum;" << endl;
  str << "}" << endl;
  
  return str.str();
}

