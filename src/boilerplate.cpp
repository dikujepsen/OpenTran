#include "StartUtil.cpp"
using namespace std;
#define LSIZE 8
cl_kernel KNearestForKernel;
cl_mem dev_ptrtrain_patterns;
cl_mem dev_ptrtest_patterns;
cl_mem dev_ptrdist_matrix;

float * hst_ptrtrain_patterns;
float * hst_ptrtest_patterns;
float * hst_ptrdist_matrix;
size_t dim;
size_t NTRAIN;
size_t NTEST;
float * hst_ptrdist_matrix_trans;
float * hst_ptrtrain_patterns_trans;

size_t hst_ptrtrain_patterns_mem_size;
size_t hst_ptrtest_patterns_mem_size;
size_t hst_ptrdist_matrix_mem_size;

size_t hst_ptrtrain_patterns_dim1;
size_t hst_ptrtrain_patterns_dim2;
size_t hst_ptrtest_patterns_dim1;
size_t hst_ptrtest_patterns_dim2;
size_t hst_ptrdist_matrix_dim1;
size_t hst_ptrdist_matrix_dim2;

size_t isFirstTime = 1;

void AllocateBuffers()
{
  hst_ptrtrain_patterns_mem_size = hst_ptrtrain_patterns_dim2 * (hst_ptrtrain_patterns_dim1 * sizeof(float));
  hst_ptrtest_patterns_mem_size = hst_ptrtest_patterns_dim2 * (hst_ptrtest_patterns_dim1 * sizeof(float));
  hst_ptrdist_matrix_mem_size = hst_ptrdist_matrix_dim2 * (hst_ptrdist_matrix_dim1 * sizeof(float));
  
  // Transposition

  hst_ptrtrain_patterns_trans = new float[hst_ptrtrain_patterns_mem_size];
  transpose<float>(
	hst_ptrtrain_patterns, hst_ptrtrain_patterns_trans, hst_ptrtrain_patterns_dim1, 
	hst_ptrtrain_patterns_dim2);
  hst_ptrdist_matrix_trans = new float[hst_ptrdist_matrix_mem_size];
  transpose<float>(
	hst_ptrdist_matrix, hst_ptrdist_matrix_trans, hst_ptrdist_matrix_dim1, 
	hst_ptrdist_matrix_dim2);
  
  // Constant Memory

  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrtrain_patterns = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrtrain_patterns_mem_size, 
	hst_ptrtrain_patterns_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtrain_patterns");
  dev_ptrtest_patterns = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrtest_patterns_mem_size, 
	hst_ptrtest_patterns, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtest_patterns");
  dev_ptrdist_matrix = clCreateBuffer(
	context, CL_MEM_COPY_HOST_PTR, hst_ptrdist_matrix_mem_size, 
	hst_ptrdist_matrix_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrdist_matrix");
}

void SetArgumentsKNearestFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(size_t), 
	(void *) &dim);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrtest_patterns_dim1);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrdist_matrix);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtrain_patterns);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrtrain_patterns_dim2);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtest_patterns);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(size_t), 
	(void *) &hst_ptrdist_matrix_dim2);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecKNearestFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t KNearestFor_global_worksize[] = {NTRAIN - 0, NTEST - 0};
  size_t KNearestFor_local_worksize[] = {LSIZE, LSIZE};
  size_t KNearestFor_global_offset[] = {0, 0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, KNearestForKernel, 2, 
	KNearestFor_global_offset, KNearestFor_global_worksize, KNearestFor_local_worksize, 
	0, NULL, &GPUExecution);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrdist_matrix, CL_TRUE, 
	0, hst_ptrdist_matrix_mem_size, hst_ptrdist_matrix, 
	1, &GPUExecution, NULL);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
}

void RunOCLKNearestForKernel(
	size_t arg_dim, float * arg_test_patterns, size_t arg_hst_ptrtest_patterns_dim1, 
	size_t arg_hst_ptrtest_patterns_dim2, float * arg_dist_matrix, size_t arg_hst_ptrdist_matrix_dim1, 
	size_t arg_hst_ptrdist_matrix_dim2, float * arg_train_patterns, size_t arg_hst_ptrtrain_patterns_dim1, 
	size_t arg_hst_ptrtrain_patterns_dim2, size_t arg_NTEST, size_t arg_NTRAIN)
{
  if (isFirstTime)
    {
      dim = arg_dim;
      hst_ptrtest_patterns = arg_test_patterns;
      hst_ptrtest_patterns_dim1 = arg_hst_ptrtest_patterns_dim1;
      hst_ptrtest_patterns_dim2 = arg_hst_ptrtest_patterns_dim2;
      hst_ptrdist_matrix = arg_dist_matrix;
      hst_ptrdist_matrix_dim1 = arg_hst_ptrdist_matrix_dim1;
      hst_ptrdist_matrix_dim2 = arg_hst_ptrdist_matrix_dim2;
      hst_ptrtrain_patterns = arg_train_patterns;
      hst_ptrtrain_patterns_dim1 = arg_hst_ptrtrain_patterns_dim1;
      hst_ptrtrain_patterns_dim2 = arg_hst_ptrtrain_patterns_dim2;
      NTEST = arg_NTEST;
      NTRAIN = arg_NTRAIN;
      StartUpGPU();
      AllocateBuffers();
      compileKernelFromFile(
	"KNearestFor", "KNearestFor.cl", &KNearestForKernel, 
	"");
      SetArgumentsKNearestFor();
    }
  ExecKNearestFor();
}
