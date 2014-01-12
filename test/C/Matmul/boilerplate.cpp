#include "../../../src/utils/StartUtil.cpp"
using namespace std;
cl_kernel matmulfunc4Kernel;
cl_mem dev_ptrA;
cl_mem dev_ptrC;
cl_mem dev_ptrB;

float * hst_ptrA;
float * hst_ptrC;
float * hst_ptrB;
size_t hA;
size_t wB;
size_t wA;

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
  str << "__kernel void matmulfunc4(" << endl;
  str << "	__global float * A, __global float * C, __global float * B" << endl;
  str << "	) {" << endl;
  str << "  __local float A_local[8*8];" << endl;
  str << "  __local float B_local[8*8];" << endl;
  str << "  unsigned li = get_local_id(1);" << endl;
  str << "  unsigned lj = get_local_id(0);" << endl;
  str << "" << endl;
  str << "  float sum = 0;" << endl;
  str << "  for (unsigned k = 0; k < wA; k+=8) {" << endl;
  str << "      A_local[(li * 8) + get_local_id(0)] = A[(get_global_id(1) * hst_ptrA_dim1) + (k + get_local_id(0))];" << endl;
  str << "      B_local[(get_local_id(1) * 8) + lj] = B[((k + get_local_id(1)) * hst_ptrB_dim1) + get_global_id(0)];" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
  str << "" << endl;
  str << "      for (unsigned kk = 0; kk < 8; kk++) {" << endl;
  str << "          sum += A_local[(li * 8) + kk] * B_local[(kk * 8) + lj];" << endl;
  str << "      }" << endl;
  str << "      barrier(CLK_LOCAL_MEM_FENCE);" << endl;
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
  std::stringstream str;
  str << "-Dhst_ptrB_dim1=" << hst_ptrB_dim1 << " ";
  str << "-Dhst_ptrA_dim1=" << hst_ptrA_dim1 << " ";
  str << "-DwA=" << wA << " ";
  str << "-Dhst_ptrC_dim1=" << hst_ptrC_dim1 << " ";
  KernelDefines = str.str();
  
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

void SetArgumentsmatmulfunc4()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrA);
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrC);
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void Execmatmulfunc4()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t matmulfunc4_global_worksize[] = {wB - 0, hA - 0};
  size_t matmulfunc4_local_worksize[] = {8, 8};
  size_t matmulfunc4_global_offset[] = {0, 0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, matmulfunc4Kernel, 2, 
	matmulfunc4_global_offset, matmulfunc4_global_worksize, matmulfunc4_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
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

void RunOCLmatmulfunc4Kernel(
	float * arg_A, size_t arg_hst_ptrA_dim1, size_t arg_hst_ptrA_dim2, 
	float * arg_C, size_t arg_hst_ptrC_dim1, size_t arg_hst_ptrC_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	size_t arg_wB, size_t arg_wA, size_t arg_hA
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
      compileKernelFromFile(
	"matmulfunc4", "matmulfunc4.cl", KernelString(), 
	false, &matmulfunc4Kernel, KernelDefines
	);
      SetArgumentsmatmulfunc4();
    }
  timer.start();
  Execmatmulfunc4();
  cout << timer.stop() << endl;
}

