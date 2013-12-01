#include "StartUtil.cpp"
using namespace std;
#define LSIZE 4
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
float * hst_ptrA_trans;
float * hst_ptrC_trans;
float * hst_ptrB_trans;

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

void AllocateBuffers()
{
  hst_ptrA_mem_size = hst_ptrA_dim2 * (hst_ptrA_dim1 * sizeof(float));
  hst_ptrC_mem_size = hst_ptrC_dim2 * (hst_ptrC_dim1 * sizeof(float));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
  
  // Transposition

  hst_ptrA_trans = new float[hst_ptrA_mem_size];
  transpose<float>(
	hst_ptrA, hst_ptrA_trans, hst_ptrA_dim1, 
	hst_ptrA_dim2);
  hst_ptrB_trans = new float[hst_ptrB_mem_size];
  transpose<float>(
	hst_ptrB, hst_ptrB_trans, hst_ptrB_dim1, 
	hst_ptrB_dim2);
  hst_ptrC_trans = new float[hst_ptrC_mem_size];
  transpose<float>(
	hst_ptrC, hst_ptrC_trans, hst_ptrC_dim1, 
	hst_ptrC_dim2);
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrA = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrA_mem_size, 
	hst_ptrA_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrA");
  dev_ptrC = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrC_mem_size, 
	hst_ptrC_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrC");
  dev_ptrB = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrB_mem_size, 
	hst_ptrB_trans, &oclErrNum);
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
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrB_dim2);
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(size_t), 
	(void *) &wA);
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrA_dim2);
  oclErrNum |= clSetKernelArg(
	matmulfunc4Kernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrC_dim2);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void Execmatmulfunc4()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t matmulfunc4_global_worksize[] = {wB, hA};
  size_t matmulfunc4_local_worksize[] = {LSIZE, LSIZE};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, matmulfunc4Kernel, 2, 
	0, matmulfunc4_global_worksize, matmulfunc4_local_worksize, 
	0, NULL, &GPUExecution);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrC, CL_TRUE, 
	0, hst_ptrC_mem_size, hst_ptrC, 
	1, &GPUExecution, NULL);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
}

void RunOCLmatmulfunc4Kernel(
	float * arg_A, size_t arg_hst_ptrA_dim1, size_t arg_hst_ptrA_dim2, 
	float * arg_C, size_t arg_hst_ptrC_dim1, size_t arg_hst_ptrC_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	size_t arg_wB, size_t arg_wA, size_t arg_hA)
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
	"matmulfunc4", "matmulfunc4.cl", &matmulfunc4Kernel, 
	"");
      SetArgumentsmatmulfunc4();
    }
  Execmatmulfunc4();
}
