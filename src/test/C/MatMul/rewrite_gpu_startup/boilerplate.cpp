#include "../../../../utils/StartUtil.cpp"
using namespace std;
cl_kernel MatMulForKernel;
cl_mem dev_ptrA;
cl_mem dev_ptrB;
cl_mem dev_ptrC;

float * hst_ptrA;
float * hst_ptrB;
float * hst_ptrC;
unsigned hA;
std::string ocl_type;
unsigned wA;
unsigned wB;

size_t hst_ptrA_mem_size;
size_t hst_ptrB_mem_size;
size_t hst_ptrC_mem_size;

size_t hst_ptrA_dim1;
size_t hst_ptrA_dim2;
size_t hst_ptrB_dim1;
size_t hst_ptrB_dim2;
size_t hst_ptrC_dim1;
size_t hst_ptrC_dim2;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string MatMulBase()
{
  std::stringstream str;
  str << "__kernel void MatMulFor(" << endl;
  str << "	__global float * A, __global float * B, __global float * C" << endl;
  str << "	) {" << endl;
  str << "  float sum = 0;" << endl;
  str << "  for (unsigned k = 0; k < wA; k++) {" << endl;
  str << "      sum += A[(get_global_id(1) * hst_ptrA_dim1) + k] * B[(k * hst_ptrB_dim1) + get_global_id(0)];" << endl;
  str << "  }" << endl;
  str << "  C[(get_global_id(1) * hst_ptrC_dim1) + get_global_id(0)] = sum;" << endl;
  str << "}" << endl;
  
  return str.str();
}


std::string MatMulPlaceInLocal()
{
  std::stringstream str;
  str << "__kernel void MatMulFor(" << endl;
  str << "	__global float * A, __global float * B, __global float * C" << endl;
  str << "	) {" << endl;
  str << "  __local float A_local[4 * 4];" << endl;
  str << "  __local float B_local[4 * 4];" << endl;
  str << "  float sum = 0;" << endl;
  str << "  for (unsigned k = 0; k < wA; k+=4) {" << endl;
  str << "      A_local[(get_local_id(1) * 4) + get_local_id(0)] = A[(get_global_id(1) * hst_ptrA_dim1) + (k + get_local_id(0))];" << endl;
  str << "      B_local[(get_local_id(1) * 4) + get_local_id(0)] = B[((k + get_local_id(1)) * hst_ptrB_dim1) + get_global_id(0)];" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "      for (unsigned kk = 0; kk < 4; kk++) {" << endl;
  str << "          sum += A_local[(get_local_id(1) * 4) + kk] * B_local[(kk * 4) + get_local_id(0)];" << endl;
  str << "      }" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "  }" << endl;
  str << "  C[(get_global_id(1) * hst_ptrC_dim1) + get_global_id(0)] = sum;" << endl;
  str << "}" << endl;
  
  return str.str();
}


std::string GetKernelCode()
{
  if (((wA - 0) % 4) == 0)
    {
      return MatMulPlaceInLocal();
    }
  else
    {
      return MatMulBase();
    }
}

void AllocateBuffers()
{
  hst_ptrA_mem_size = hst_ptrA_dim2 * (hst_ptrA_dim1 * sizeof(float));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
  hst_ptrC_mem_size = hst_ptrC_dim2 * (hst_ptrC_dim1 * sizeof(float));
  
  // Transposition
  
  // Constant Memory
  
  // Defines for the kernel
  std::stringstream str;
  str << "-Dhst_ptrA_dim1=" << hst_ptrA_dim1 << " ";
  str << "-Dhst_ptrB_dim1=" << hst_ptrB_dim1 << " ";
  str << "-Dhst_ptrC_dim1=" << hst_ptrC_dim1 << " ";
  str << "-DwA=" << wA << " ";
  KernelDefines = str.str();
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrA = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrA_mem_size, 
	hst_ptrA, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrA");
  dev_ptrB = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrB_mem_size, 
	hst_ptrB, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrB");
  dev_ptrC = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrC_mem_size, 
	hst_ptrC, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrC");
}

void SetArgumentsMatMulFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrA);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrC);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecMatMulFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t MatMulFor_global_worksize[] = {wB - 0, hA - 0};
  size_t MatMulFor_local_worksize[] = {4, 4};
  size_t MatMulFor_global_offset[] = {0, 0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, MatMulForKernel, 2, 
	MatMulFor_global_offset, MatMulFor_global_worksize, MatMulFor_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrC, CL_TRUE, 
	0, hst_ptrC_mem_size, hst_ptrC, 
	1, &GPUExecution, NULL
	);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLMatMulForKernel(
	float * arg_A, size_t arg_hst_ptrA_dim1, size_t arg_hst_ptrA_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	float * arg_C, size_t arg_hst_ptrC_dim1, size_t arg_hst_ptrC_dim2, 
	unsigned arg_hA, std::string arg_ocl_type, unsigned arg_wA, 
	unsigned arg_wB)
{
  if (isFirstTime)
    {
      hst_ptrA = arg_A;
      hst_ptrA_dim1 = arg_hst_ptrA_dim1;
      hst_ptrA_dim2 = arg_hst_ptrA_dim2;
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      hst_ptrC = arg_C;
      hst_ptrC_dim1 = arg_hst_ptrC_dim1;
      hst_ptrC_dim2 = arg_hst_ptrC_dim2;
      hA = arg_hA;
      ocl_type = arg_ocl_type;
      wA = arg_wA;
      wB = arg_wB;
      StartUpOCL(ocl_type);
      AllocateBuffers();
      cout << "$Defines " << KernelDefines << endl;
      compileKernel(
	"MatMulFor", "MatMulFor.cl", GetKernelCode(), 
	false, &MatMulForKernel, KernelDefines
	);
      SetArgumentsMatMulFor();
    }
  timer.start();
  ExecMatMulFor();
  cout << "$Time " << timer.stop() << endl;
}

