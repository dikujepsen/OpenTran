#include "../../../utils/StartUtil.cpp"
using namespace std;

class OCLNBodyTask
{
  cl_kernel NBodyForKernel;
    cl_mem dev_ptrForces;
  cl_mem dev_ptrMas;
  cl_mem dev_ptrPos;

    float * hst_ptrForces;
  float * hst_ptrMas;
  size_t N;
  std::string ocl_type;
  float * hst_ptrPos;

    size_t hst_ptrForces_mem_size;
  size_t hst_ptrMas_mem_size;
  size_t hst_ptrPos_mem_size;

    size_t hst_ptrForces_dim1;
  size_t hst_ptrForces_dim2;
  size_t hst_ptrMas_dim1;
  size_t hst_ptrPos_dim1;
  size_t hst_ptrPos_dim2;

    size_t isFirstTime = 1;
  std::string KernelDefines = "";
  Stopwatch timer;


public:
  OCLNBodyTask()
  {
    isFirstTime = 1;
    KernelDefines = 1;
  }

  void RunOCLNBodyForKernel(
	float * arg_Forces, size_t arg_hst_ptrForces_dim1, size_t arg_hst_ptrForces_dim2, 
	float * arg_Mas, size_t arg_hst_ptrMas_dim1, size_t arg_N, 
	float * arg_Pos, size_t arg_hst_ptrPos_dim1, size_t arg_hst_ptrPos_dim2, 
	std::string arg_ocl_type)
  {
    if (isFirstTime)
      {
        hst_ptrForces = arg_Forces;
        hst_ptrForces_dim1 = arg_hst_ptrForces_dim1;
        hst_ptrForces_dim2 = arg_hst_ptrForces_dim2;
        hst_ptrMas = arg_Mas;
        hst_ptrMas_dim1 = arg_hst_ptrMas_dim1;
        N = arg_N;
        hst_ptrPos = arg_Pos;
        hst_ptrPos_dim1 = arg_hst_ptrPos_dim1;
        hst_ptrPos_dim2 = arg_hst_ptrPos_dim2;
        ocl_type = arg_ocl_type;
        StartUpOCL(ocl_type);
        AllocateBuffers();
        cout << "$Defines " << KernelDefines << endl;
        compileKernel(
	"NBodyFor", "NBodyFor.cl", GetKernelCode(), 
	false, &NBodyForKernel, KernelDefines
	);
        SetArgumentsNBodyFor();
      }
    timer.start();
    ExecNBodyFor();
    cout << "$Time " << timer.stop() << endl;
  }


private:
  std::string NBodyBase()
  {
    std::stringstream str;
    str << "__kernel void NBodyFor(" << endl;
    str << "	__global float * Forces, __global float * Mas, __global float * Pos" << endl;
    str << "	) {" << endl;
    str << "  float f_x = 0;" << endl;
    str << "  float f_y = 0;" << endl;
    str << "  for (unsigned j = 0; j < N; j++) {" << endl;
    str << "      float a_x = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
    str << "      float a_y = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
    str << "      float a_m = Mas[get_global_id(0)];" << endl;
    str << "      float b_x = Pos[(0 * hst_ptrPos_dim1) + j];" << endl;
    str << "      float b_y = Pos[(1 * hst_ptrPos_dim1) + j];" << endl;
    str << "      float b_m = Mas[j];" << endl;
    str << "      float r_x = b_x - a_x;" << endl;
    str << "      float r_y = b_y - a_y;" << endl;
    str << "      float d = (r_x * r_x) + (r_y * r_y);" << endl;
    str << "      float deno = sqrt((d * d) * d) + (get_global_id(0) == j);" << endl;
    str << "      deno = ((a_m * b_m) / deno) * (get_global_id(0) != j);" << endl;
    str << "      f_x += deno * r_x;" << endl;
    str << "      f_y += deno * r_y;" << endl;
    str << "  }" << endl;
    str << "  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;" << endl;
    str << "  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string NBodyPlaceInReg()
  {
    std::stringstream str;
    str << "__kernel void NBodyFor(" << endl;
    str << "	__global float * Forces, __global float * Mas, __global float * Pos" << endl;
    str << "	) {" << endl;
    str << "  float Mas0_reg = Mas[get_global_id(0)];" << endl;
    str << "  float Pos0_reg = Pos[(0 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
    str << "  float Pos1_reg = Pos[(1 * hst_ptrPos_dim1) + get_global_id(0)];" << endl;
    str << "  float f_x = 0;" << endl;
    str << "  float f_y = 0;" << endl;
    str << "  for (unsigned j = 0; j < N; j++) {" << endl;
    str << "      float a_x = Pos0_reg;" << endl;
    str << "      float a_y = Pos1_reg;" << endl;
    str << "      float a_m = Mas0_reg;" << endl;
    str << "      float b_x = Pos[(0 * hst_ptrPos_dim1) + j];" << endl;
    str << "      float b_y = Pos[(1 * hst_ptrPos_dim1) + j];" << endl;
    str << "      float b_m = Mas[j];" << endl;
    str << "      float r_x = b_x - a_x;" << endl;
    str << "      float r_y = b_y - a_y;" << endl;
    str << "      float d = (r_x * r_x) + (r_y * r_y);" << endl;
    str << "      float deno = sqrt((d * d) * d) + (get_global_id(0) == j);" << endl;
    str << "      deno = ((a_m * b_m) / deno) * (get_global_id(0) != j);" << endl;
    str << "      f_x += deno * r_x;" << endl;
    str << "      f_y += deno * r_y;" << endl;
    str << "  }" << endl;
    str << "  Forces[(0 * hst_ptrForces_dim1) + get_global_id(0)] = f_x;" << endl;
    str << "  Forces[(1 * hst_ptrForces_dim1) + get_global_id(0)] = f_y;" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string GetKernelCode()
  {
    if ((1 * 3) < 40)
      {
        return NBodyPlaceInReg();
      }
    else
      {
        return NBodyBase();
      }
  }

  void AllocateBuffers()
  {
        hst_ptrForces_mem_size = hst_ptrForces_dim2 * (hst_ptrForces_dim1 * sizeof(float));
    hst_ptrMas_mem_size = hst_ptrMas_dim1 * sizeof(float);
    hst_ptrPos_mem_size = hst_ptrPos_dim2 * (hst_ptrPos_dim1 * sizeof(float));

        // Transposition

        // Constant Memory

        // Defines for the kernel
    std::stringstream str;
    str << "-DN=" << N << " ";
    str << "-Dhst_ptrForces_dim1=" << hst_ptrForces_dim1 << " ";
    str << "-Dhst_ptrPos_dim1=" << hst_ptrPos_dim1 << " ";
    KernelDefines = str.str();

        cl_int oclErrNum = CL_SUCCESS;

    dev_ptrForces = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrForces_mem_size, 
	hst_ptrForces, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrForces");
    dev_ptrMas = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrMas_mem_size, 
	hst_ptrMas, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrMas");
    dev_ptrPos = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrPos_mem_size, 
	hst_ptrPos, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrPos");
  }

  void SetArgumentsNBodyFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    int counter = 0;
    oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrForces);
    oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrMas);
    oclErrNum |= clSetKernelArg(
	NBodyForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrPos);
    oclCheckErr(
	oclErrNum, "clSetKernelArg");
  }

  void ExecNBodyFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    cl_event GPUExecution;
    size_t NBodyFor_global_worksize[] = {N - 0};
    size_t NBodyFor_local_worksize[] = {16};
    size_t NBodyFor_global_offset[] = {0};
    oclErrNum = clEnqueueNDRangeKernel(
	command_queue, NBodyForKernel, 1, 
	NBodyFor_global_offset, NBodyFor_global_worksize, NBodyFor_local_worksize, 
	0, NULL, &GPUExecution
	);
    oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
    oclErrNum = clFinish(command_queue);
    oclCheckErr(
	oclErrNum, "clFinish");
    oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrForces, CL_TRUE, 
	0, hst_ptrForces_mem_size, hst_ptrForces, 
	1, &GPUExecution, NULL
	);
    oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clFinish(command_queue);
    oclCheckErr(
	oclErrNum, "clFinish");
  }


}
