#include "StartUtil.cpp"
using namespace std;
#define LSIZE 8
cl_kernel NBody2ForKernel;
cl_mem dev_ptrMas;
cl_mem dev_ptrPos;
cl_mem dev_ptrForces;
cl_mem dev_ptrConstantMasPos;

float * hst_ptrMas;
float * hst_ptrPos;
float * hst_ptrForces;
size_t N;
float * hst_ptrConstantMasPos;

size_t hst_ptrMas_mem_size;
size_t hst_ptrConstantMasPos_mem_size;
size_t hst_ptrPos_mem_size;
size_t hst_ptrForces_mem_size;

size_t hst_ptrMas_dim1;
size_t hst_ptrPos_dim1;
size_t hst_ptrPos_dim2;
size_t hst_ptrForces_dim1;
size_t hst_ptrForces_dim2;

size_t isFirstTime = 1;
std::string KernelDefines = "";
std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void NBody2For(" << endl;
  str << "	unsigned hst_ptrForces_dim1, __global float * Mas, __global float * Pos, " << endl;
  str << "	unsigned N, __constant float * ConstantMasPos, __global float * Forces, " << endl;
  str << "	unsigned hst_ptrPos_dim1) {" << endl;
  str << "  float Mas0_reg = Mas[get_global_id(0)];" << endl;
  str << "  float Pos0_reg = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float Pos1_reg = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "" << endl;
  str << "" << endl;
  str << "  float f_x = 0;" << endl;
  str << "  float f_y = 0;" << endl;
  str << "  for (unsigned j = 0; j < N; j++) {" << endl;
  str << "      float b_m = Mas0_reg * ConstantMasPos[(3 * j) + 0];" << endl;
  str << "      float a_x = Pos0_reg;" << endl;
  str << "      float a_y = Pos1_reg;" << endl;
  str << "      float r_x = ConstantMasPos[(3 * j) + 1] - a_x;" << endl;
  str << "      float r_y = ConstantMasPos[(3 * j) + 2] - a_y;" << endl;
  str << "      float d = (r_x * r_x) + (r_y * r_y);" << endl;
  str << "      float deno = (sqrt((d * d) * d)) + (get_global_id(0) == j);" << endl;
  str << "      deno = (b_m / deno) * (get_global_id(0) != j);" << endl;
  str << "      f_x += deno * r_x;" << endl;
  str << "      f_y += deno * r_y;" << endl;
  str << "  }" << endl;
  str << "  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;" << endl;
  str << "  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;" << endl;
  str << "}" << endl;
  
  return str.str();
}


void AllocateBuffers()
{
  hst_ptrMas_mem_size = hst_ptrMas_dim1 * sizeof(float);
  hst_ptrPos_mem_size = hst_ptrPos_dim2 * (hst_ptrPos_dim1 * sizeof(float));
  hst_ptrForces_mem_size = hst_ptrForces_dim2 * (hst_ptrForces_dim1 * sizeof(float));
  hst_ptrConstantMasPos_mem_size = hst_ptrMas_mem_size + hst_ptrPos_mem_size;
  
  // Transposition
  
  // Constant Memory
  hst_ptrConstantMasPos = new float[hst_ptrConstantMasPos_mem_size];
  for (unsigned j = 0; j < N; j++)
    {
      hst_ptrConstantMasPos[(3 * j) + 0] = hst_ptrMas[j];
      hst_ptrConstantMasPos[(3 * j) + 1] = hst_ptrPos[(0 * hst_ptrPos_dim1) + j];
      hst_ptrConstantMasPos[(3 * j) + 2] = hst_ptrPos[(1 * hst_ptrPos_dim1) + j];
    }
  
  
  // Defines for the kernel
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrMas = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrMas_mem_size, 
	hst_ptrMas, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrMas");
  dev_ptrConstantMasPos = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR, hst_ptrConstantMasPos_mem_size, 
	hst_ptrConstantMasPos, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrConstantMasPos");
  dev_ptrPos = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrPos_mem_size, 
	hst_ptrPos, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrPos");
  dev_ptrForces = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrForces_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrForces");
}

void SetArgumentsNBody2For()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrForces_dim1);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrMas);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrPos);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(unsigned), 
	(void *) &N);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrConstantMasPos);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrForces);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrPos_dim1);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecNBody2For()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t NBody2For_global_worksize[] = {N - 0};
  size_t NBody2For_local_worksize[] = {LSIZE};
  size_t NBody2For_global_offset[] = {0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, NBody2ForKernel, 1, 
	NBody2For_global_offset, NBody2For_global_worksize, NBody2For_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrForces, CL_TRUE, 
	0, hst_ptrForces_mem_size, hst_ptrForces, 
	1, &GPUExecution, NULL
	);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLNBody2ForKernel(
	float * arg_Mas, size_t arg_hst_ptrMas_dim1, float * arg_Pos, 
	size_t arg_hst_ptrPos_dim1, size_t arg_hst_ptrPos_dim2, float * arg_Forces, 
	size_t arg_hst_ptrForces_dim1, size_t arg_hst_ptrForces_dim2, size_t arg_N
	)
{
  if (isFirstTime)
    {
      hst_ptrMas = arg_Mas;
      hst_ptrMas_dim1 = arg_hst_ptrMas_dim1;
      hst_ptrPos = arg_Pos;
      hst_ptrPos_dim1 = arg_hst_ptrPos_dim1;
      hst_ptrPos_dim2 = arg_hst_ptrPos_dim2;
      hst_ptrForces = arg_Forces;
      hst_ptrForces_dim1 = arg_hst_ptrForces_dim1;
      hst_ptrForces_dim2 = arg_hst_ptrForces_dim2;
      N = arg_N;
      StartUpGPU();
      AllocateBuffers();
      compileKernelFromFile(
	"NBody2For", "NBody2For.cl", KernelString(), 
	false, &NBody2ForKernel, KernelDefines
	);
      SetArgumentsNBody2For();
    }
  ExecNBody2For();
}

