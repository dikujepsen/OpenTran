#include "StartUtil.cpp"
using namespace std;
#define LSIZE 8
cl_kernel KNearestFor2Kernel;
cl_mem dev_ptrtrain_patterns;
cl_mem dev_ptrtest_patterns;
cl_mem dev_ptrdist_matrix;

float * hst_ptrtrain_patterns;
float * hst_ptrtest_patterns;
float * hst_ptrdist_matrix;
size_t dim;
size_t NTEST;
size_t NTRAIN;
float * hst_ptrtest_patterns_trans;

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
std::string KernelDefines = "";

void AllocateBuffers()
{
  hst_ptrtrain_patterns_mem_size = hst_ptrtrain_patterns_dim2 * (hst_ptrtrain_patterns_dim1 * sizeof(float));
  hst_ptrtest_patterns_mem_size = hst_ptrtest_patterns_dim2 * (hst_ptrtest_patterns_dim1 * sizeof(float));
  hst_ptrdist_matrix_mem_size = hst_ptrdist_matrix_dim2 * (hst_ptrdist_matrix_dim1 * sizeof(float));
  
  // Transposition

  hst_ptrtest_patterns_trans = new float[hst_ptrtest_patterns_mem_size];
  transpose<float>(
	hst_ptrtest_patterns, hst_ptrtest_patterns_trans, hst_ptrtest_patterns_dim1, 
	hst_ptrtest_patterns_dim2);
  
  // Constant Memory

  
  // Defines for the kernel

  std::stringstream str;
  str << "-Ddim=" << dim << " ";
  str << "-DNTRAIN=" << NTRAIN << " ";
  str << "-Dhst_ptrtrain_patterns_dim1=" << hst_ptrtrain_patterns_dim1 << " ";
  str << "-Dhst_ptrtest_patterns_dim1=" << hst_ptrtest_patterns_dim2 << " ";
  str << "-Dhst_ptrdist_matrix_dim1=" << hst_ptrdist_matrix_dim1 << " ";
  KernelDefines = str.str();
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrtrain_patterns = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR, hst_ptrtrain_patterns_mem_size, 
	hst_ptrtrain_patterns, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtrain_patterns");
  dev_ptrtest_patterns = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR, hst_ptrtest_patterns_mem_size, 
	hst_ptrtest_patterns_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtest_patterns");
  dev_ptrdist_matrix = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrdist_matrix_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrdist_matrix");
}

void SetArgumentsKNearestFor2()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	KNearestFor2Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrdist_matrix);
  oclErrNum |= clSetKernelArg(
	KNearestFor2Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtrain_patterns);
  oclErrNum |= clSetKernelArg(
	KNearestFor2Kernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtest_patterns);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecKNearestFor2()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t KNearestFor2_global_worksize[] = {NTEST - 0};
  size_t KNearestFor2_local_worksize[] = {LSIZE};
  size_t KNearestFor2_global_offset[] = {0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, KNearestFor2Kernel, 1, 
	KNearestFor2_global_offset, KNearestFor2_global_worksize, KNearestFor2_local_worksize, 
	0, NULL, &GPUExecution);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrdist_matrix, CL_TRUE, 
	0, hst_ptrdist_matrix_mem_size, hst_ptrdist_matrix, 
	1, &GPUExecution, NULL);
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLKNearestFor2Kernel(
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
	"KNearestFor2", "KNearestFor2.cl", &KNearestFor2Kernel, 
	KernelDefines);
      SetArgumentsKNearestFor2();
    }
  ExecKNearestFor2();
}
