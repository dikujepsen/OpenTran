#include "../../../src/utils/StartUtil.cpp"
using namespace std;
cl_kernel JacobiForKernel;
cl_mem dev_ptrX2;
cl_mem dev_ptrB;
cl_mem dev_ptrX1;

unknown * hst_ptrX2;
unknown * hst_ptrB;
unknown * hst_ptrX1;
unknown wB;
unknown wA;

size_t hst_ptrX2_mem_size;
size_t hst_ptrX1_mem_size;
size_t hst_ptrB_mem_size;

size_t hst_ptrX2_dim1;
size_t hst_ptrX2_dim2;
size_t hst_ptrB_dim1;
size_t hst_ptrB_dim2;
size_t hst_ptrX1_dim1;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void JacobiFor(" << endl;
  str << "	__global unknown * B, unsigned hst_ptrB_dim1, unknown wA, " << endl;
  str << "	unsigned hst_ptrX2_dim1, __global unknown * X2, __global unknown * X1" << endl;
  str << "	) {" << endl;
  str << "  X2[(get_global_id(1) * hst_ptrX2_dim1) + get_global_id(0)] = (-0.25) * ((B[(get_global_id(1) * hst_ptrB_dim1) + get_global_id(0)] - (X1[((get_global_id(1) - 1) * wA) + get_global_id(0)] + X1[((get_global_id(1) + 1) * wA) + get_global_id(0)])) - (X1[(get_global_id(1) * wA) + (get_global_id(0) - 1)] + X1[(get_global_id(1) * wA) + (get_global_id(0) + 1)]));" << endl;
  str << "}" << endl;
  
  return str.str();
}


void AllocateBuffers()
{
  hst_ptrX2_mem_size = hst_ptrX2_dim2 * (hst_ptrX2_dim1 * sizeof(unknown));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(unknown));
  hst_ptrX1_mem_size = hst_ptrX1_dim1 * sizeof(unknown);
  
  // Transposition
  
  // Constant Memory
  
  // Defines for the kernel
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrX2 = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrX2_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrX2");
  dev_ptrX1 = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrX1_mem_size, 
	hst_ptrX1, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrX1");
  dev_ptrB = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrB_mem_size, 
	hst_ptrB, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrB");
}

void SetArgumentsJacobiFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrB_dim1);
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(unknown), 
	(void *) &wA);
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(unsigned), 
	(void *) &hst_ptrX2_dim1);
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX2);
  oclErrNum |= clSetKernelArg(
	JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX1);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecJacobiFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t JacobiFor_global_worksize[] = {wB - 1, wB - 1};
  size_t JacobiFor_local_worksize[] = {16, 16};
  size_t JacobiFor_global_offset[] = {1, 1};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, JacobiForKernel, 2, 
	JacobiFor_global_offset, JacobiFor_global_worksize, JacobiFor_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrX2, CL_TRUE, 
	0, hst_ptrX2_mem_size, hst_ptrX2, 
	1, &GPUExecution, NULL
	);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLJacobiForKernel(
	unknown * arg_X2, size_t arg_hst_ptrX2_dim1, size_t arg_hst_ptrX2_dim2, 
	unknown * arg_X1, size_t arg_hst_ptrX1_dim1, unknown * arg_B, 
	size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, unknown arg_wB, 
	unknown arg_wA)
{
  if (isFirstTime)
    {
      hst_ptrX2 = arg_X2;
      hst_ptrX2_dim1 = arg_hst_ptrX2_dim1;
      hst_ptrX2_dim2 = arg_hst_ptrX2_dim2;
      hst_ptrX1 = arg_X1;
      hst_ptrX1_dim1 = arg_hst_ptrX1_dim1;
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      wB = arg_wB;
      wA = arg_wA;
      StartUpGPU();
      AllocateBuffers();
      cout << KernelDefines << endl;
      compileKernelFromFile(
	"JacobiFor", "JacobiFor.cl", KernelString(), 
	false, &JacobiForKernel, KernelDefines
	);
      SetArgumentsJacobiFor();
    }
  timer.start();
  ExecJacobiFor();
  cout << timer.stop() << endl;
}

