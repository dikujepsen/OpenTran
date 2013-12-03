#include "StartUtil.cpp"
using namespace std;
#define LSIZE 4
cl_kernel JacobiKernel;
cl_mem dev_ptrX2;
cl_mem dev_ptrX1;
cl_mem dev_ptrB;

float * hst_ptrX2;
float * hst_ptrX1;
float * hst_ptrB;
unsigned wB;
unsigned wA;

size_t hst_ptrX2_mem_size;
size_t hst_ptrX1_mem_size;
size_t hst_ptrB_mem_size;

size_t hst_ptrX2_dim1;
size_t hst_ptrX2_dim2;
size_t hst_ptrX1_dim1;
size_t hst_ptrX1_dim2;
size_t hst_ptrB_dim1;
size_t hst_ptrB_dim2;

size_t isFirstTime = 1;

void AllocateBuffers()
{
  hst_ptrX2_mem_size = hst_ptrX2_dim2 * (hst_ptrX2_dim1 * sizeof(float));
  hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
  hst_ptrX1_mem_size = hst_ptrX1_dim2 * (hst_ptrX1_dim1 * sizeof(float));
  
  // Transposition

  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrX2 = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrX2_mem_size, 
	hst_ptrX2, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrX2");
  dev_ptrB = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrB_mem_size, 
	hst_ptrB, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrB");
  dev_ptrX1 = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrX1_mem_size, 
	hst_ptrX1, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrX1");
}

void SetArgumentsJacobi()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrB_dim1);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(unsigned), 
	(void *) &wA);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrX1_dim1);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrX2_dim1);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX2);
  oclErrNum |= clSetKernelArg(
	JacobiKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX1);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecJacobi()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t Jacobi_global_worksize[] = {wB - 1, wB - 1};
  size_t Jacobi_local_worksize[] = {LSIZE, LSIZE};
  size_t Jacobi_global_offset[] = {1, 1};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, JacobiKernel, 2, 
	Jacobi_global_offset, Jacobi_global_worksize, Jacobi_local_worksize, 
	0, NULL, &GPUExecution);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrX2, CL_TRUE, 
	0, hst_ptrX2_mem_size, hst_ptrX2, 
	1, &GPUExecution, NULL);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
}

void RunOCLJacobiKernel(
	float * arg_X2, size_t arg_hst_ptrX2_dim1, size_t arg_hst_ptrX2_dim2, 
	float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	float * arg_X1, size_t arg_hst_ptrX1_dim1, size_t arg_hst_ptrX1_dim2, 
	unsigned arg_wB, unsigned arg_wA)
{
  if (isFirstTime)
    {
      hst_ptrX2 = arg_X2;
      hst_ptrX2_dim1 = arg_hst_ptrX2_dim1;
      hst_ptrX2_dim2 = arg_hst_ptrX2_dim2;
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      hst_ptrX1 = arg_X1;
      hst_ptrX1_dim1 = arg_hst_ptrX1_dim1;
      hst_ptrX1_dim2 = arg_hst_ptrX1_dim2;
      wB = arg_wB;
      wA = arg_wA;
      StartUpGPU();
      AllocateBuffers();
      compileKernelFromFile(
	"Jacobi", "Jacobi.cl", &JacobiKernel, 
	"");
      SetArgumentsJacobi();
    }
  ExecJacobi();
}
