#include "../../../utils/StartUtil.cpp"
using namespace std;
cl_kernel KNearestForKernel;
cl_mem dev_ptrtrain_patterns;
cl_mem dev_ptrtest_patterns;
cl_mem dev_ptrdist_matrix;

float * hst_ptrtrain_patterns;
float * hst_ptrtest_patterns;
float * hst_ptrdist_matrix;
unsigned dim;
unsigned NTEST;
unsigned NTRAIN;
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
std::string KernelDefines = "";
Stopwatch timer;

std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void KNearestFor(" << endl;
  str << "	__global float * dist_matrix, __global float * train_patterns, __global float * test_patterns" << endl;
  str << "	) {" << endl;
  str << "  float d = 0.0;" << endl;
  str << "  for (unsigned k = 0; k < dim; k++) {" << endl;
  str << "      float tmp = test_patterns[(get_global_id(1) * hst_ptrtest_patterns_dim1) + k] - train_patterns[(k * hst_ptrtrain_patterns_dim1) + get_global_id(0)];" << endl;
  str << "      d += tmp * tmp;" << endl;
  str << "  }" << endl;
  str << "  dist_matrix[(get_global_id(1) * hst_ptrdist_matrix_dim1) + get_global_id(0)] = d;" << endl;
  str << "}" << endl;
  
  return str.str();
}


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
  
  // Constant Memory
  
  // Defines for the kernel
  std::stringstream str;
  str << "-Ddim=" << dim << " ";
  str << "-Dhst_ptrtest_patterns_dim1=" << hst_ptrtest_patterns_dim1 << " ";
  str << "-Dhst_ptrtrain_patterns_dim1=" << hst_ptrtrain_patterns_dim2 << " ";
  str << "-Dhst_ptrdist_matrix_dim1=" << hst_ptrdist_matrix_dim2 << " ";
  KernelDefines = str.str();
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrtrain_patterns = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrtrain_patterns_mem_size, 
	hst_ptrtrain_patterns_trans, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtrain_patterns");
  dev_ptrtest_patterns = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrtest_patterns_mem_size, 
	hst_ptrtest_patterns, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrtest_patterns");
  dev_ptrdist_matrix = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrdist_matrix_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrdist_matrix");
}

void SetArgumentsKNearestFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrdist_matrix);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtrain_patterns);
  oclErrNum |= clSetKernelArg(
	KNearestForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrtest_patterns);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecKNearestFor()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t KNearestFor_global_worksize[] = {NTRAIN - 0, NTEST - 0};
  size_t KNearestFor_local_worksize[] = {16, 16};
  size_t KNearestFor_global_offset[] = {0, 0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, KNearestForKernel, 2, 
	KNearestFor_global_offset, KNearestFor_global_worksize, KNearestFor_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
  oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
}

void RunOCLKNearestForKernel(
	unsigned arg_dim, float * arg_test_patterns, size_t arg_hst_ptrtest_patterns_dim1, 
	size_t arg_hst_ptrtest_patterns_dim2, float * arg_dist_matrix, size_t arg_hst_ptrdist_matrix_dim1, 
	size_t arg_hst_ptrdist_matrix_dim2, float * arg_train_patterns, size_t arg_hst_ptrtrain_patterns_dim1, 
	size_t arg_hst_ptrtrain_patterns_dim2, unsigned arg_NTEST, unsigned arg_NTRAIN
	)
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
      cout << "$Defines " << KernelDefines << endl;
      compileKernel(
	"KNearestFor", "KNearestFor.cl", KernelString(), 
	false, &KNearestForKernel, KernelDefines
	);
      SetArgumentsKNearestFor();
    }
  timer.start();
  ExecKNearestFor();
  cout << "$Time " << timer.stop() << endl;
}

