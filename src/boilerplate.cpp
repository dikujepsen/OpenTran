#include "StartUtil.cpp"
using namespace std;
#define LSIZE 4
cl_kernel NBodyForKernel;
cl_mem dev_ptrMas;
cl_mem dev_ptrPos;
cl_mem dev_ptrForces;
cl_mem dev_ptr_constantPos;

float * hst_ptrMas;
float * hst_ptrPos;
float * hst_ptrForces;
size_t N;
float * hst_ptr_constantPos;

size_t hst_ptr_constantPos_mem_size;
size_t hst_ptrMas_mem_size;
size_t hst_ptrPos_mem_size;
size_t hst_ptrForces_mem_size;

size_t hst_ptrMas_dim1;
size_t hst_ptrPos_dim1;
size_t hst_ptrPos_dim2;
size_t hst_ptrForces_dim1;
size_t hst_ptrForces_dim2;

size_t isFirstTime = 1;

void AllocateBuffers()
{
  hst_ptrMas_mem_size = hst_ptrMas_dim1 * sizeof(float);
  hst_ptrPos_mem_size = hst_ptrPos_dim2 * (hst_ptrPos_dim1 * sizeof(float));
  hst_ptrForces_mem_size = hst_ptrForces_dim2 * (hst_ptrForces_dim1 * sizeof(float));
  hst_ptr_constantPos_mem_size = hst_ptrPos_mem_size * sizeof(float);
  
  // Transposition

  
  // Constant Memory

  hst_ptr_constantPos = new float[hst_ptr_constantPos_mem_size];
  size_t counter = 0;
  for (unsigned j = 0; j < N; j++)
    {
      hst_ptr_constantPos[2*j-2] = hst_ptrPos[(0 * hst_ptrPos_dim1) + j];
      hst_ptr_constantPos[2*j-1] = hst_ptrPos[(1 * hst_ptrPos_dim1) + j];
    }
  
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptr_constantPos = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptr_constantPos_mem_size, 
	hst_ptr_constantPos, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptr_constantPos");
  dev_ptrMas = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrMas_mem_size, 
	hst_ptrMas, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrMas");
  dev_ptrPos = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrPos_mem_size, 
	hst_ptrPos, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrPos");
  dev_ptrForces = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrForces_mem_size, 
	hst_ptrForces, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrForces");
}

void SetArgumentsNBodyFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrForces_dim1);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptr_constantPos);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrMas);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrPos);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(size_t), 
	(void *) &N);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrForces);
  oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrPos_dim1);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecNBodyFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t NBodyFor_global_worksize[] = {N - 0};
  size_t NBodyFor_local_worksize[] = {LSIZE};
  size_t NBodyFor_global_offset[] = {0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, NBodyForKernel, 1, 
	NBodyFor_global_offset, NBodyFor_global_worksize, NBodyFor_local_worksize, 
	0, NULL, &GPUExecution);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrForces, CL_TRUE, 
	0, hst_ptrForces_mem_size, hst_ptrForces, 
	1, &GPUExecution, NULL);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
}

void RunOCLNBodyForKernel(
	float * arg_Mas, size_t arg_hst_ptrMas_dim1, float * arg_Pos, 
	size_t arg_hst_ptrPos_dim1, size_t arg_hst_ptrPos_dim2, float * arg_Forces, 
	size_t arg_hst_ptrForces_dim1, size_t arg_hst_ptrForces_dim2, size_t arg_N)
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
	"NBodyFor", "NBodyFor.cl", &NBodyForKernel, 
	"");
      SetArgumentsNBodyFor();
    }
  ExecNBodyFor();
}
