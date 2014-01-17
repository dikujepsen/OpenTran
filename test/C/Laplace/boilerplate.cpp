#include "../../../src/utils/StartUtil.cpp"
using namespace std;
cl_kernel LaplaceForKernel;
cl_mem dev_ptrindex;
cl_mem dev_ptrlevel_int;
cl_mem dev_ptrlevel;
cl_mem dev_ptrlcl_q;
cl_mem dev_ptrresult;
cl_mem dev_ptrlcl_q_inv;
cl_mem dev_ptralpha;
cl_mem dev_ptrlambda;

float * hst_ptrindex;
float * hst_ptrlevel_int;
float * hst_ptrlevel;
float * hst_ptrlcl_q;
float * hst_ptrresult;
float * hst_ptrlcl_q_inv;
float * hst_ptralpha;
float * hst_ptrlambda;
size_t dim;
size_t storagesize;
float * hst_ptrlevel_trans;
float * hst_ptrlevel_int_trans;
float * hst_ptrindex_trans;

size_t hst_ptrindex_mem_size;
size_t hst_ptrlevel_int_mem_size;
size_t hst_ptrlevel_mem_size;
size_t hst_ptrlcl_q_mem_size;
size_t hst_ptrresult_mem_size;
size_t hst_ptrlcl_q_inv_mem_size;
size_t hst_ptralpha_mem_size;
size_t hst_ptrlambda_mem_size;

size_t hst_ptrindex_dim1;
size_t hst_ptrindex_dim2;
size_t hst_ptrlevel_int_dim1;
size_t hst_ptrlevel_int_dim2;
size_t hst_ptrlevel_dim1;
size_t hst_ptrlevel_dim2;
size_t hst_ptrlcl_q_dim1;
size_t hst_ptrresult_dim1;
size_t hst_ptrlcl_q_inv_dim1;
size_t hst_ptralpha_dim1;
size_t hst_ptrlambda_dim1;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string KernelString()
{
  std::stringstream str;
  str << "#include \"LaplaceIncludes.hpp\"" << endl;
  str << "__kernel void LaplaceFor(" << endl;
  str << "	__global float * level_int, __global float * lambda, __global float * level, " << endl;
  str << "	__global float * lcl_q, __global float * index, __global float * result, " << endl;
  str << "	__global float * lcl_q_inv, __global float * alpha) {" << endl;
  str << "  unsigned li = get_local_id(0);" << endl;
  str << "  __local float alpha_local[1*8];" << endl;
  str << "  float index_reg[dim];" << endl;
  str << "  float level_int_reg[dim];" << endl;
  str << "  float level_reg[dim];" << endl;
    for (unsigned d = 0; d < dim; d++) {
  str << "      index_reg[" << d << "] = index[(" << d << " * hst_ptrindex_dim1) + get_global_id(0)];" << endl;
  str << "      level_int_reg[" << d << "] = level_int[(" << d << " * hst_ptrlevel_int_dim1) + get_global_id(0)];" << endl;
  str << "      level_reg[" << d << "] = level[(" << d << " * hst_ptrlevel_dim1) + get_global_id(0)];" << endl;
    }
  str << "  float sub = 0.0;" << endl;
  str << "  float gradient_temp[dim];" << endl;
  str << "  float dot_temp[dim];" << endl;
  str << "  for (unsigned j = 0; j < storagesize; j+=8) {" << endl;
  str << "      alpha_local[get_local_id(0)] = alpha[j + get_local_id(0)];" << endl;
  str << "      for (unsigned jj = 0; jj < 8; jj++) {" << endl;
            for (unsigned d = 0; d < dim; d+=dim) {
  str << "              float level_int_i;" << endl;
  str << "              float level_i;" << endl;
  str << "              float index_i;" << endl;
  str << "              float level_int_j;" << endl;
  str << "              float index_j;" << endl;
  str << "              float level_j;" << endl;
                for (unsigned dd = 0; dd < dim; dd++) {
  str << "                  level_i = level_reg[" << d << " + " << dd << "];" << endl;
  str << "                  level_j = level[((" << d << " + " << dd << ") * hst_ptrlevel_dim1) + j + jj];" << endl;
  str << "                  level_int_i = level_int_reg[" << d << " + " << dd << "];" << endl;
  str << "                  level_int_j = level_int[((" << d << " + " << dd << ") * hst_ptrlevel_int_dim1) + j + jj];" << endl;
  str << "                  index_i = index_reg[" << d << " + " << dd << "];" << endl;
  str << "                  index_j = index[((" << d << " + " << dd << ") * hst_ptrindex_dim1) + j + jj];" << endl;
  str << "                  gradient_temp[" << d << " + " << dd << "] = gradient(" << endl;
  str << "	level_i, index_i, level_j, " << endl;
  str << "	index_j, lcl_q_inv[" << d << " + " << dd << "]);" << endl;
  str << "                  dot_temp[" << d << " + " << dd << "] = l2dot(" << endl;
  str << "	level_i, level_j, index_i, " << endl;
  str << "	index_j, level_int_i, level_int_j, " << endl;
  str << "	lcl_q[" << d << " + " << dd << "]);" << endl;
                }
            }
  str << "          float alphatemp = alpha_local[jj];" << endl;
            for (unsigned d_outer = 0; d_outer < dim; d_outer+=dim) {
  str << "              float element;" << endl;
                for (unsigned d_outerd_outer = 0; d_outerd_outer < dim; d_outerd_outer++) {
  str << "                  element = alphatemp;" << endl;
                    for (unsigned d_inner = 0; d_inner < dim; d_inner+=dim) {
                        for (unsigned d_innerd_inner = 0; d_innerd_inner < dim; d_innerd_inner++) {
  str << "                          element *= (dot_temp[" << d_inner << " + " << d_innerd_inner << "] * ((" << d_outer << " + " << d_outerd_outer << ") != (" << d_inner << " + " << d_innerd_inner << "))) + (gradient_temp[" << d_inner << " + " << d_innerd_inner << "] * ((" << d_outer << " + " << d_outerd_outer << ") == (" << d_inner << " + " << d_innerd_inner << ")));" << endl;
                        }
                    }
  str << "                  sub += lambda[" << d_outer << " + " << d_outerd_outer << "] * element;" << endl;
                }
            }
  str << "      }" << endl;
  str << "  }" << endl;
  str << "  result[get_global_id(0)] = sub;" << endl;
  str << "}" << endl;
  
  return str.str();
}


void AllocateBuffers()
{
  hst_ptrindex_mem_size = hst_ptrindex_dim2 * (hst_ptrindex_dim1 * sizeof(float));
  hst_ptrlevel_int_mem_size = hst_ptrlevel_int_dim2 * (hst_ptrlevel_int_dim1 * sizeof(float));
  hst_ptrlevel_mem_size = hst_ptrlevel_dim2 * (hst_ptrlevel_dim1 * sizeof(float));
  hst_ptrlcl_q_mem_size = hst_ptrlcl_q_dim1 * sizeof(float);
  hst_ptrresult_mem_size = hst_ptrresult_dim1 * sizeof(float);
  hst_ptrlcl_q_inv_mem_size = hst_ptrlcl_q_inv_dim1 * sizeof(float);
  hst_ptralpha_mem_size = hst_ptralpha_dim1 * sizeof(float);
  hst_ptrlambda_mem_size = hst_ptrlambda_dim1 * sizeof(float);
  
  // Transposition
  hst_ptrlevel_trans = new float[hst_ptrlevel_mem_size];
  transpose<float>(
	hst_ptrlevel, hst_ptrlevel_trans, hst_ptrlevel_dim1, 
	hst_ptrlevel_dim2);
  hst_ptrlevel_int_trans = new float[hst_ptrlevel_int_mem_size];
  transpose<float>(
	hst_ptrlevel_int, hst_ptrlevel_int_trans, hst_ptrlevel_int_dim1, 
	hst_ptrlevel_int_dim2);
  hst_ptrindex_trans = new float[hst_ptrindex_mem_size];
  transpose<float>(
	hst_ptrindex, hst_ptrindex_trans, hst_ptrindex_dim1, 
	hst_ptrindex_dim2);
  
  // Constant Memory
  
  // Defines for the kernel
  std::stringstream str;
  str << "-Ddim=" << dim << " ";
  str << "-Dhst_ptrlevel_dim1=" << hst_ptrlevel_dim2 << " ";
  str << "-Dhst_ptrindex_dim1=" << hst_ptrindex_dim2 << " ";
  str << "-Dstoragesize=" << storagesize << " ";
  str << "-Dhst_ptrlevel_int_dim1=" << hst_ptrlevel_int_dim2 << " ";
  KernelDefines = str.str();
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrindex = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrindex_mem_size, 
	hst_ptrindex_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrindex");
  dev_ptrlevel_int = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlevel_int_mem_size, 
	hst_ptrlevel_int_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlevel_int");
  dev_ptrlevel = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlevel_mem_size, 
	hst_ptrlevel_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlevel");
  dev_ptrlcl_q = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlcl_q_mem_size, 
	hst_ptrlcl_q, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlcl_q");
  dev_ptrresult = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrresult_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrresult");
  dev_ptrlcl_q_inv = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlcl_q_inv_mem_size, 
	hst_ptrlcl_q_inv, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlcl_q_inv");
  dev_ptralpha = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptralpha_mem_size, 
	hst_ptralpha, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptralpha");
  dev_ptrlambda = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlambda_mem_size, 
	hst_ptrlambda, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlambda");
}

void SetArgumentsLaplaceFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlevel_int);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlambda);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlevel);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlcl_q);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrindex);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrresult);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlcl_q_inv);
  oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptralpha);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecLaplaceFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t LaplaceFor_global_worksize[] = {storagesize - 0};
  size_t LaplaceFor_local_worksize[] = {8};
  size_t LaplaceFor_global_offset[] = {0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, LaplaceForKernel, 1, 
	LaplaceFor_global_offset, LaplaceFor_global_worksize, LaplaceFor_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrresult, CL_TRUE, 
	0, hst_ptrresult_mem_size, hst_ptrresult, 
	1, &GPUExecution, NULL
	);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLLaplaceForKernel(
	size_t arg_dim, float * arg_level_int, size_t arg_hst_ptrlevel_int_dim1, 
	size_t arg_hst_ptrlevel_int_dim2, float * arg_index, size_t arg_hst_ptrindex_dim1, 
	size_t arg_hst_ptrindex_dim2, float * arg_lcl_q, size_t arg_hst_ptrlcl_q_dim1, 
	float * arg_level, size_t arg_hst_ptrlevel_dim1, size_t arg_hst_ptrlevel_dim2, 
	float * arg_result, size_t arg_hst_ptrresult_dim1, float * arg_lcl_q_inv, 
	size_t arg_hst_ptrlcl_q_inv_dim1, float * arg_alpha, size_t arg_hst_ptralpha_dim1, 
	size_t arg_storagesize, float * arg_lambda, size_t arg_hst_ptrlambda_dim1
	)
{
  if (isFirstTime)
    {
      dim = arg_dim;
      hst_ptrlevel_int = arg_level_int;
      hst_ptrlevel_int_dim1 = arg_hst_ptrlevel_int_dim1;
      hst_ptrlevel_int_dim2 = arg_hst_ptrlevel_int_dim2;
      hst_ptrindex = arg_index;
      hst_ptrindex_dim1 = arg_hst_ptrindex_dim1;
      hst_ptrindex_dim2 = arg_hst_ptrindex_dim2;
      hst_ptrlcl_q = arg_lcl_q;
      hst_ptrlcl_q_dim1 = arg_hst_ptrlcl_q_dim1;
      hst_ptrlevel = arg_level;
      hst_ptrlevel_dim1 = arg_hst_ptrlevel_dim1;
      hst_ptrlevel_dim2 = arg_hst_ptrlevel_dim2;
      hst_ptrresult = arg_result;
      hst_ptrresult_dim1 = arg_hst_ptrresult_dim1;
      hst_ptrlcl_q_inv = arg_lcl_q_inv;
      hst_ptrlcl_q_inv_dim1 = arg_hst_ptrlcl_q_inv_dim1;
      hst_ptralpha = arg_alpha;
      hst_ptralpha_dim1 = arg_hst_ptralpha_dim1;
      storagesize = arg_storagesize;
      hst_ptrlambda = arg_lambda;
      hst_ptrlambda_dim1 = arg_hst_ptrlambda_dim1;
      StartUpGPU();
      AllocateBuffers();
      compileKernelFromFile(
	"LaplaceFor", "LaplaceFor.cl", KernelString(), 
	false, &LaplaceForKernel, KernelDefines
	);
      SetArgumentsLaplaceFor();
    }
  timer.start();
  ExecLaplaceFor();
  cout << timer.stop() << endl;
}

