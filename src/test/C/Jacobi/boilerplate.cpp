#include "../../../utils/StartUtil.cpp"
#include "../../../utils/helper.hpp"
using namespace std;

class OCLJacobiTask
{
  cl_kernel JacobiForKernel;
  cl_mem dev_ptrB;
  cl_mem dev_ptrX1;
  cl_mem dev_ptrX2;

  float * hst_ptrB;
  std::string ocl_type;
  unsigned wB;
  float * hst_ptrX1;
  float * hst_ptrX2;

  size_t hst_ptrB_mem_size;
  size_t hst_ptrX1_mem_size;
  size_t hst_ptrX2_mem_size;

  size_t hst_ptrB_dim1;
  size_t hst_ptrB_dim2;
  size_t hst_ptrX1_dim1;
  size_t hst_ptrX1_dim2;
  size_t hst_ptrX2_dim1;
  size_t hst_ptrX2_dim2;

  size_t isFirstTime;
  std::string KernelDefines;
  Stopwatch timer;
  OCLContext * ocl_context;


public:
  OCLJacobiTask()
  {
    isFirstTime = 1;
    KernelDefines = "";
  }

  void RunOCLJacobiForKernel(float * arg_B, size_t arg_hst_ptrB_dim1, size_t arg_hst_ptrB_dim2, 
	float * arg_X1, size_t arg_hst_ptrX1_dim1, size_t arg_hst_ptrX1_dim2, 
	float * arg_X2, size_t arg_hst_ptrX2_dim1, size_t arg_hst_ptrX2_dim2, 
	std::string arg_ocl_type, unsigned arg_wB)
  {
    if (isFirstTime)
    {
      hst_ptrB = arg_B;
      hst_ptrB_dim1 = arg_hst_ptrB_dim1;
      hst_ptrB_dim2 = arg_hst_ptrB_dim2;
      hst_ptrX1 = arg_X1;
      hst_ptrX1_dim1 = arg_hst_ptrX1_dim1;
      hst_ptrX1_dim2 = arg_hst_ptrX1_dim2;
      hst_ptrX2 = arg_X2;
      hst_ptrX2_dim1 = arg_hst_ptrX2_dim1;
      hst_ptrX2_dim2 = arg_hst_ptrX2_dim2;
      ocl_type = arg_ocl_type;
      wB = arg_wB;
      ocl_context = new OCLContext();
      ocl_context->StartUpOCL(ocl_type);
      AllocateBuffers();
      cout << "$Defines " << KernelDefines << endl;
      ocl_context->compileKernel("JacobiFor", "JacobiFor.cl", GetKernelCode(), 
	false, &JacobiForKernel, KernelDefines
	);
      SetArgumentsJacobiFor();
    }
    timer.start();
    ExecJacobiFor();
    cout << "$Time " << timer.stop() << endl;
  }


private:
  std::string JacobiBase()
  {
    std::stringstream str;
    str << "__kernel void JacobiFor(" << endl;
    str << "  __global float * B, __global float * X1, __global float * X2" << endl;
    str << "  ) {" << endl;
    str << "  __local float X1_local[6 * 6];" << endl;
    str << "  unsigned li = get_local_id(1) + 1;" << endl;
    str << "  unsigned lj = get_local_id(0) + 1;" << endl;
    str << "  X1_local[((li - 1) * 4) + lj] = X1[((get_global_id(1) - 1) * hst_ptrX1_dim1) + get_global_id(0)];" << endl;
    str << "  X1_local[((li + 1) * 4) + lj] = X1[((get_global_id(1) + 1) * hst_ptrX1_dim1) + get_global_id(0)];" << endl;
    str << "  X1_local[(li * 4) + (lj - 1)] = X1[(get_global_id(1) * hst_ptrX1_dim1) + (get_global_id(0) - 1)];" << endl;
    str << "  X1_local[(li * 4) + (lj + 1)] = X1[(get_global_id(1) * hst_ptrX1_dim1) + (get_global_id(0) + 1)];" << endl;
    str << "  barrier(CLK_LOCAL_MEM_FENCE);" << endl;
    str << "  X2[(get_global_id(1) * hst_ptrX2_dim1) + get_global_id(0)] = (-0.25) * ((B[(get_global_id(1) * hst_ptrB_dim1) + get_global_id(0)] - (X1_local[((li - 1) * 4) + lj] + X1_local[((li + 1) * 4) + lj])) - (X1_local[(li * 4) + (lj - 1)] + X1_local[(li * 4) + (lj + 1)]));" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string GetKernelCode()
  {
    return JacobiBase();
  }

  void AllocateBuffers()
  {
    hst_ptrB_mem_size = hst_ptrB_dim2 * (hst_ptrB_dim1 * sizeof(float));
    hst_ptrX1_mem_size = hst_ptrX1_dim2 * (hst_ptrX1_dim1 * sizeof(float));
    hst_ptrX2_mem_size = hst_ptrX2_dim2 * (hst_ptrX2_dim1 * sizeof(float));

    // Transposition

    // Constant Memory

    // Defines for the kernel
    std::stringstream str;
    str << "-Dhst_ptrB_dim1=" << hst_ptrB_dim1 << " ";
    str << "-Dhst_ptrX1_dim1=" << hst_ptrX1_dim1 << " ";
    str << "-Dhst_ptrX2_dim1=" << hst_ptrX2_dim1 << " ";
    KernelDefines = str.str();

    cl_int oclErrNum = CL_SUCCESS;

    dev_ptrB = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrB_mem_size, 
	hst_ptrB, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrB");
    dev_ptrX1 = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrX1_mem_size, 
	hst_ptrX1, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrX1");
    dev_ptrX2 = clCreateBuffer(ocl_context->getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrX2_mem_size, 
	hst_ptrX2, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrX2");
  }

  void SetArgumentsJacobiFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    int counter = 0;
    oclErrNum |= clSetKernelArg(JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrB);
    oclErrNum |= clSetKernelArg(JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX1);
    oclErrNum |= clSetKernelArg(JacobiForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrX2);
    helper::oclCheckErr(oclErrNum, "clSetKernelArg");
  }

  void ExecJacobiFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    cl_event GPUExecution;
    size_t JacobiFor_global_worksize[] = {wB - 1, wB - 1};
    size_t JacobiFor_local_worksize[] = {4, 4};
    size_t JacobiFor_global_offset[] = {1, 1};
    oclErrNum = clEnqueueNDRangeKernel(ocl_context->getCommandQueue(), JacobiForKernel, 2, 
	JacobiFor_global_offset, JacobiFor_global_worksize, JacobiFor_local_worksize, 
	0, NULL, &GPUExecution
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueNDRangeKernel");
    oclErrNum = clFinish(ocl_context->getCommandQueue());
    helper::oclCheckErr(oclErrNum, "clFinish");
    oclErrNum = clEnqueueReadBuffer(ocl_context->getCommandQueue(), dev_ptrX2, CL_TRUE, 
	0, hst_ptrX2_mem_size, hst_ptrX2, 
	1, &GPUExecution, NULL
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clFinish(ocl_context->getCommandQueue());
    helper::oclCheckErr(oclErrNum, "clFinish");
  }


};
