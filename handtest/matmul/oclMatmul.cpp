#include "../StartUtil.cpp"
using namespace std;

#define LSIZE 4

cl_kernel matmulKernel[NUMDEVS];

cl_mem dev_ptrA[NUMDEVS];
cl_mem dev_ptrB[NUMDEVS];
cl_mem dev_ptrC[NUMDEVS];

float* hst_ptrA;
float* hst_ptrB;
float* hst_ptrC;

size_t hst_ptrA_mem_size = 0;
size_t hst_ptrB_mem_size = 0;
size_t hst_ptrC_mem_size = 0;

size_t hst_ptrA_dim1 = 0;
size_t hst_ptrB_dim1 = 0;
size_t hst_ptrC_dim1 = 0;

size_t hst_ptrA_dim2 = 0;
size_t hst_ptrB_dim2 = 0;
size_t hst_ptrC_dim2 = 0;

size_t dev_ptrA_dim1 = 0;
size_t dev_ptrB_dim1 = 0;
size_t dev_ptrC_dim1 = 0;

size_t dev_ptrA_dim2 = 0;
size_t dev_ptrB_dim2 = 0;
size_t dev_ptrC_dim2 = 0;

size_t isFirstTime = 1;

void AllocateBuffers() {

  hst_ptrA_mem_size = hst_ptrA_dim1 * hst_ptrA_dim2 * sizeof(float);
  hst_ptrB_mem_size = hst_ptrB_dim1 * hst_ptrB_dim2 * sizeof(float);
  hst_ptrC_mem_size = hst_ptrC_dim1 * hst_ptrC_dim2 * sizeof(float);

  dev_ptrA_dim1 = hst_ptrA_dim1;
  dev_ptrB_dim1 = hst_ptrB_dim1;
  dev_ptrC_dim1 = hst_ptrC_dim1;
  dev_ptrA_dim2 = hst_ptrA_dim2;
  dev_ptrB_dim2 = hst_ptrB_dim2;
  dev_ptrC_dim2 = hst_ptrC_dim2;

  size_t dev_ptrA_mem_size = hst_ptrA_dim1 * hst_ptrA_dim2 * sizeof(float);
  size_t dev_ptrB_mem_size = hst_ptrB_dim1 * hst_ptrB_dim2 * sizeof(float);
  size_t dev_ptrC_mem_size = hst_ptrC_dim1 * hst_ptrC_dim2 * sizeof(float);

  cl_int ciErrNum = CL_SUCCESS;
  int i = 0;
  dev_ptrA[i] = clCreateBuffer(context,
			       CL_MEM_COPY_HOST_PTR,
			       dev_ptrA_mem_size,
			       hst_ptrA,
			       &ciErrNum);
  oclCheckErr(ciErrNum, "clCreateBuffer ptrA");

  dev_ptrB[i] = clCreateBuffer(context,
			       CL_MEM_COPY_HOST_PTR,
			       dev_ptrB_mem_size,
			       hst_ptrB,
			       &ciErrNum);
  oclCheckErr(ciErrNum, "clCreateBuffer ptrB");

  dev_ptrC[i] = clCreateBuffer(context,
			       CL_MEM_COPY_HOST_PTR,
			       dev_ptrC_mem_size,
			       hst_ptrC,
			       &ciErrNum);
  oclCheckErr(ciErrNum, "clCreateBuffer ptrC");
  

  
}

void SetArgumentsMatmul() {

  cl_int ciErrNum = CL_SUCCESS;
  int counter = 0;
  int i = 0;
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_mem), (void *) &dev_ptrA[i]);
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_mem), (void *) &dev_ptrB[i]);
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_mem), (void *) &dev_ptrC[i]);
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_uint), (void *) &dev_ptrA_dim2);
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_uint), (void *) &dev_ptrA_dim1);
  ciErrNum |= clSetKernelArg(matmulKernel[i], counter++, sizeof(cl_uint), (void *) &dev_ptrB_dim1);
  oclCheckErr(ciErrNum, "clSetKernelArg");
  
}

void ExecMatmulKernel() {
  cl_int ciErrNum = CL_SUCCESS;
  cl_event GPUExecution[NUMDEVS];

  size_t matmul_global_worksize[] = {dev_ptrC_dim1, dev_ptrC_dim2};
  size_t matmul_local_worksize[] = {LSIZE,LSIZE};
  int i = 0;
  ciErrNum = clEnqueueNDRangeKernel(command_queue[i],
				    matmulKernel[i], 2, 0,
				    matmul_global_worksize,
				    matmul_local_worksize,
				    0, NULL, &GPUExecution[i]);
  oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel");
  ciErrNum = clEnqueueReadBuffer(command_queue[i],
				 dev_ptrC[i],
				 CL_TRUE, 0,
				 hst_ptrC_mem_size,
				 hst_ptrC, 1, &GPUExecution[i],
				 NULL);
  oclCheckErr(ciErrNum, "clEnqueueReadBuffer");
}

void RunOCLMatmulKernel(float* arg_hst_ptrA, 	size_t arg_hst_ptrA_dim1,
			size_t arg_hst_ptrA_dim2,
			float* arg_hst_ptrB, 	size_t arg_hst_ptrB_dim1,
			size_t arg_hst_ptrB_dim2,
			float* arg_hst_ptrC, 	size_t arg_hst_ptrC_dim1,
			size_t arg_hst_ptrC_dim2
			) {
  

  if (isFirstTime) {
    hst_ptrA      = arg_hst_ptrA;
    hst_ptrA_dim1 = arg_hst_ptrA_dim1;
    hst_ptrA_dim2 = arg_hst_ptrA_dim2;
    hst_ptrB 	  = arg_hst_ptrB;
    hst_ptrB_dim1 = arg_hst_ptrB_dim1;
    hst_ptrB_dim2 = arg_hst_ptrB_dim2;
    hst_ptrC 	  = arg_hst_ptrC;
    hst_ptrC_dim1 = arg_hst_ptrC_dim1;
    hst_ptrC_dim2 = arg_hst_ptrC_dim2;

    StartUpGPU();
    AllocateBuffers();
    compileKernelFromFile("matmul", "matmul.cl", matmulKernel, "");
    SetArgumentsMatmul();
  }
  ExecMatmulKernel();

}


