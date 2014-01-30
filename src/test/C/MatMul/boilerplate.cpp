#include "../../../utils/StartUtil.cpp"
using namespace std;
cl_kernel MatMulForKernel;
cl_mem dev_ptrA;
cl_mem dev_ptrC;
cl_mem dev_ptrB;

float * hst_ptrA;
float * hst_ptrC;
float * hst_ptrB;
unsigned hA;
unsigned wB;
unsigned wA;

size_t hst_ptrA_mem_size;
size_t hst_ptrC_mem_size;
size_t hst_ptrB_mem_size;

size_t hst_ptrA_dim1;
size_t hst_ptrA_dim2;
size_t hst_ptrC_dim1;
size_t hst_ptrC_dim2;
size_t hst_ptrB_dim1;
size_t hst_ptrB_dim2;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void MatMulFor(" << endl;
  str << "	__global float * A, __global float * C, __global float * B, " << endl;
  str << "	unsigned hst_ptrB_dim1, unsigned wA, unsigned hst_ptrA_dim1, " << endl;
  str << "	unsigned hst_ptrC_dim1) {" << endl;
  str << "  float sum = 0;" << endl;
  str << "  for (unsigned k = 0; k < wA; k++) {" << endl;
  str << "      sum += A[(get_global_id(1) * hst_ptrA_dim1) + k] * B[(k * hst_ptrB_dim1) + get_global_id(0)];" << endl;
  str << "  }" << endl;
  str << "  C[(get_global_id(1) * hst_ptrC_dim1) + get_global_id(0)] = sum;" << endl;
  str << "}" << endl;
  
  return str.str();
}


void AllocateBuffers()
{
  hst_ptrA_mem_size = hst_ptrA_dim2 * (hst_ptrA_dim1 * sizeof(float));
  hst_ptrC_mem_size = hst_ptrC_dim2 * (hst_ptrC_dim1 * sizeof(float));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
  
  // Transposition
  
  // Constant Memory
  
  // Defines for the kernel
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrA = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrA_mem_size, 
	hst_ptrA, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrA");
  dev_ptrC = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrC_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrC");
  dev_ptrB = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrB_mem_size, 
	hst_ptrB, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrB");
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
	(void *) &dev_ptrC);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrB_dim1);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(unsigned), 
	(void *) &wA);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrA_dim1);
  oclErrNum |= clSetKernelArg(
	MatMulForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrC_dim1);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecMatMulFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t MatMulFor_global_worksize[] = {wB - 0, hA - 0};
  size_t MatMulFor_local_worksize[] = {16, 16};
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
	float * arg_C, size_t arg_hst_ptrC_dim1, size_t arg_hst_ptrC_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	unsigned arg_wB, unsigned arg_wA, unsigned arg_hA
	)
{
  if (isFirstTime)
    {
      hst_ptrA = arg_A;
      hst_ptrA_dim1 = arg_hst_ptrA_dim1;
      hst_ptrA_dim2 = arg_hst_ptrA_dim2;
      hst_ptrC = arg_C;
      hst_ptrC_dim1 = arg_hst_ptrC_dim1;
      hst_ptrC_dim2 = arg_hst_ptrC_dim2;
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      wB = arg_wB;
      wA = arg_wA;
      hA = arg_hA;
      StartUpGPU();
      AllocateBuffers();
      cout << "$Defines " << KernelDefines << endl;
      compileKernel(
	"MatMulFor", "MatMulFor.cl", KernelString(), 
	false, &MatMulForKernel, KernelDefines
	);
      SetArgumentsMatMulFor();
    }
  timer.start();
  ExecMatMulFor();
  cout << "$Time " << timer.stop() << endl;
}

