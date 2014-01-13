#include "../../../src/utils/StartUtil.cpp"
using namespace std;
cl_kernel NBody2ForKernel;
cl_mem dev_ptrMas;
cl_mem dev_ptrForces_y;
cl_mem dev_ptrPos;
cl_mem dev_ptrForces_x;

float * hst_ptrMas;
float * hst_ptrForces_y;
float * hst_ptrPos;
float * hst_ptrForces_x;
size_t N;

size_t hst_ptrMas_mem_size;
size_t hst_ptrForces_y_mem_size;
size_t hst_ptrPos_mem_size;
size_t hst_ptrForces_x_mem_size;

size_t hst_ptrMas_dim1;
size_t hst_ptrForces_y_dim1;
size_t hst_ptrForces_y_dim2;
size_t hst_ptrPos_dim1;
size_t hst_ptrPos_dim2;
size_t hst_ptrForces_x_dim1;
size_t hst_ptrForces_x_dim2;

size_t isFirstTime = 1;
std::string KernelDefines = "";
Stopwatch timer;

std::string KernelString()
{
  std::stringstream str;
  str << "__kernel void NBody2For(" << endl;
  str << "	__global float * Mas, __global float * Pos, __global float * Forces_y, " << endl;
  str << "	__global float * Forces_x) {" << endl;
  str << "  float b_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float b_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
  str << "  float b_m = Mas[get_global_id(0)];" << endl;
  str << "  float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(1)];" << endl;
  str << "  float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(1)];" << endl;
  str << "  float a_m = Mas[get_global_id(1)];" << endl;
  str << "  float r_x = b_x - a_x;" << endl;
  str << "  float r_y = b_y - a_y;" << endl;
  str << "  float d = (r_x * r_x) + (r_y * r_y);" << endl;
  str << "  float deno = (sqrt((d * d) * d)) + (get_global_id(1) == get_global_id(0));" << endl;
  str << "  deno = ((a_m * b_m) / deno) * (get_global_id(1) != get_global_id(0));" << endl;
  str << "  Forces_x[(get_global_id(1) * hst_ptrForces_x_dim1) + get_global_id(0)] = deno * r_x;" << endl;
  str << "  Forces_y[(get_global_id(1) * hst_ptrForces_y_dim1) + get_global_id(0)] = deno * r_y;" << endl;
  str << "}" << endl;
  
  return str.str();
}


void AllocateBuffers()
{
  hst_ptrMas_mem_size = hst_ptrMas_dim1 * sizeof(float);
  hst_ptrForces_y_mem_size = hst_ptrForces_y_dim2 * (hst_ptrForces_y_dim1 * sizeof(float));
  hst_ptrPos_mem_size = hst_ptrPos_dim2 * (hst_ptrPos_dim1 * sizeof(float));
  hst_ptrForces_x_mem_size = hst_ptrForces_x_dim2 * (hst_ptrForces_x_dim1 * sizeof(float));
  
  // Transposition
  
  // Constant Memory
  
  // Defines for the kernel
  std::stringstream str;
  str << "-Dhst_ptrForces_x_dim1=" << hst_ptrForces_x_dim1 << " ";
  str << "-Dhst_ptrForces_y_dim1=" << hst_ptrForces_y_dim1 << " ";
  str << "-Dhst_ptrPos_dim1=" << hst_ptrPos_dim1 << " ";
  KernelDefines = str.str();
  
  cl_int oclErrNum = CL_SUCCESS;
  
  dev_ptrMas = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrMas_mem_size, 
	hst_ptrMas, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrMas");
  dev_ptrForces_y = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrForces_y_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrForces_y");
  dev_ptrPos = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrPos_mem_size, 
	hst_ptrPos, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrPos");
  dev_ptrForces_x = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY, hst_ptrForces_x_mem_size, 
	NULL, &oclErrNum);
  oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrForces_x");
}

void SetArgumentsNBody2For()
{
  cl_int oclErrNum = CL_SUCCESS;
  int counter = 0;
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrMas);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrPos);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrForces_y);
  oclErrNum |= clSetKernelArg(
	NBody2ForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrForces_x);
  oclCheckErr(
	oclErrNum, "clSetKernelArg");
}

void ExecNBody2For()
{
  cl_int oclErrNum = CL_SUCCESS;
  cl_event GPUExecution;
  size_t NBody2For_global_worksize[] = {N - 0, N/256 - 0};
  size_t NBody2For_local_worksize[] = {256, 1};
  size_t NBody2For_global_offset[] = {0, 0};
  oclErrNum = clEnqueueNDRangeKernel(
	command_queue, NBody2ForKernel, 2, 
	NBody2For_global_offset, NBody2For_global_worksize, NBody2For_local_worksize, 
	0, NULL, &GPUExecution
	);
  oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
  // oclErrNum = clEnqueueReadBuffer(
  // 	command_queue, dev_ptrForces_y, CL_TRUE, 
  // 	0, hst_ptrForces_y_mem_size, hst_ptrForces_y, 
  // 	1, &GPUExecution, NULL
  // 	);
  // oclErrNum = clEnqueueReadBuffer(
  // 	command_queue, dev_ptrForces_x, CL_TRUE, 
  // 	0, hst_ptrForces_x_mem_size, hst_ptrForces_x, 
  // 	1, &GPUExecution, NULL
  // 	);
  // oclCheckErr(
  // 	oclErrNum, "clEnqueueReadBuffer");
  oclErrNum = clFinish(command_queue);
  oclCheckErr(
	oclErrNum, "clFinish");
}

void RunOCLNBody2ForKernel(
	float * arg_Mas, size_t arg_hst_ptrMas_dim1, float * arg_Forces_y, 
	size_t arg_hst_ptrForces_y_dim1, size_t arg_hst_ptrForces_y_dim2, float * arg_Pos, 
	size_t arg_hst_ptrPos_dim1, size_t arg_hst_ptrPos_dim2, float * arg_Forces_x, 
	size_t arg_hst_ptrForces_x_dim1, size_t arg_hst_ptrForces_x_dim2, size_t arg_N
	)
{
  if (isFirstTime)
    {
      hst_ptrMas = arg_Mas;
      hst_ptrMas_dim1 = arg_hst_ptrMas_dim1;
      hst_ptrForces_y = arg_Forces_y;
      hst_ptrForces_y_dim1 = arg_hst_ptrForces_y_dim1;
      hst_ptrForces_y_dim2 = arg_hst_ptrForces_y_dim2;
      hst_ptrPos = arg_Pos;
      hst_ptrPos_dim1 = arg_hst_ptrPos_dim1;
      hst_ptrPos_dim2 = arg_hst_ptrPos_dim2;
      hst_ptrForces_x = arg_Forces_x;
      hst_ptrForces_x_dim1 = arg_hst_ptrForces_x_dim1;
      hst_ptrForces_x_dim2 = arg_hst_ptrForces_x_dim2;
      N = arg_N;
      StartUpGPU();
      AllocateBuffers();
      compileKernelFromFile(
	"NBody2For", "NBody2For.cl", KernelString(), 
	true, &NBody2ForKernel, KernelDefines
	);
      SetArgumentsNBody2For();
    }
  timer.start();
  ExecNBody2For();
  cout << timer.stop() << endl;
}

