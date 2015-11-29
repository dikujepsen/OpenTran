#include "../../../utils/StartUtil.cpp"
using namespace std;
cl_kernel JacobiForKernel;
cl_mem dev_ptrX2;
cl_mem dev_ptrB;
cl_mem dev_ptrX1;

float * hst_ptrX2;
float * hst_ptrB;
float * hst_ptrX1;
unsigned wB;
unsigned wA;

size_t hst_ptrX2_mem_size;
size_t hst_ptrX1_mem_size;
size_t hst_ptrB_mem_size;

size_t hst_ptrX2_dim1;
size_t hst_ptrX2_dim2;
size_t hst_ptrB_dim1;
size_t hst_ptrB_dim2;
size_t hst_ptrX1_dim1;
size_t hst_ptrX1_dim2;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string JacobiBase()
{
  std::stringstream str;
  str << "__kernel void JacobiFor(" << endl;
  str << "	__global float * B, __global float * X2, __global float * X1" << endl;
  str << "	) {" << endl;
  str << "  __local float X1_local[18*18];" << endl;
  str << "  unsigned li = get_local_id(1) + 1;" << endl;
  str << "  unsigned lj = get_local_id(0) + 1;" << endl;
  str << "  X1_local[((li - 1) * 16) + lj] = X1[((get_global_id(1) - 1) * hst_ptrX1_dim1) + get_global_id(0)];" << endl;
  str << "  X1_local[((li + 1) * 16) + lj] = X1[((get_global_id(1) + 1) * hst_ptrX1_dim1) + get_global_id(0)];" << endl;
  str << "  X1_local[(li * 16) + (lj - 1)] = X1[(get_global_id(1) * hst_ptrX1_dim1) + (get_global_id(0) - 1)];" << endl;
  str << "  X1_local[(li * 16) + (lj + 1)] = X1[(get_global_id(1) * hst_ptrX1_dim1) + (get_global_id(0) + 1)];" << endl;
  str << "  barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "  X2[(get_global_id(1) * hst_ptrX2_dim1) + get_global_id(0)] = (-0.25) * ((B[(get_global_id(1) * hst_ptrB_dim1) + get_global_id(0)] - (X1_local[((li - 1) * 16) + lj] + X1_local[((li + 1) * 16) + lj])) - (X1_local[(li * 16) + (lj - 1)] + X1_local[(li * 16) + (lj + 1)]));" << endl;
  str << "}" << endl;
  
  return str.str();
}


std::string GetKernelCode()
{
  return  JacobiBase();
}

void AllocateBuffers()
{
  hst_ptrX2_mem_size = hst_ptrX2_dim2 * (hst_ptrX2_dim1 * sizeof(float));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
  hst_ptrX1_mem_size = hst_ptrX1_dim2 * (hst_ptrX1_dim1 * sizeof(float));
  
  // Transposition
  
  // Constant Memory
  
  // Defines for the kernel
  std::stringstream str;
  str << "-Dhst_ptrB_dim1=" << hst_ptrB_dim1 << " ";
  str << "-DwA=" << wA << " ";
  str << "-Dhst_ptrX1_dim1=" << hst_ptrX1_dim1 << " ";
  str << "-Dhst_ptrX2_dim1=" << hst_ptrX2_dim1 << " ";
  KernelDefines = str.str();
  
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
}

void RunOCLJacobiForKernel(
	float * arg_X2, size_t arg_hst_ptrX2_dim1, size_t arg_hst_ptrX2_dim2, 
	float * arg_X1, size_t arg_hst_ptrX1_dim1, size_t arg_hst_ptrX1_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	unsigned arg_wB, unsigned arg_wA)
{
  if (isFirstTime)
    {
      hst_ptrX2 = arg_X2;
      hst_ptrX2_dim1 = arg_hst_ptrX2_dim1;
      hst_ptrX2_dim2 = arg_hst_ptrX2_dim2;
      hst_ptrX1 = arg_X1;
      hst_ptrX1_dim1 = arg_hst_ptrX1_dim1;
      hst_ptrX1_dim2 = arg_hst_ptrX1_dim2;
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      wB = arg_wB;
      wA = arg_wA;
      StartUpGPU();
      AllocateBuffers();
      cout << "$Defines " << KernelDefines << endl;
      compileKernel(
	"JacobiFor", "JacobiFor.cl", GetKernelCode(), 
	false, &JacobiForKernel, KernelDefines
	);
      SetArgumentsJacobiFor();
    }
  timer.start();
  ExecJacobiFor();
  cout << "$Time " << timer.stop() << endl;
}

