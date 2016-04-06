#include "../../../utils/StartUtil.cpp"
using namespace std;

class OCLLaplaceTask
{
  cl_kernel LaplaceForKernel;
    cl_mem dev_ptralpha;
  cl_mem dev_ptrindex;
  cl_mem dev_ptrlambda;
  cl_mem dev_ptrlcl_q;
  cl_mem dev_ptrlcl_q_inv;
  cl_mem dev_ptrlevel;
  cl_mem dev_ptrlevel_int;
  cl_mem dev_ptrresult;

    double * hst_ptralpha;
  size_t dim;
  double * hst_ptrindex;
  double * hst_ptrlambda;
  double * hst_ptrlcl_q;
  double * hst_ptrlcl_q_inv;
  double * hst_ptrlevel;
  double * hst_ptrlevel_int;
  std::string ocl_type;
  double * hst_ptrresult;
  size_t storagesize;
  double * hst_ptrindex_trans;
  double * hst_ptrlevel_int_trans;
  double * hst_ptrlevel_trans;

    size_t hst_ptralpha_mem_size;
  size_t hst_ptrindex_mem_size;
  size_t hst_ptrlambda_mem_size;
  size_t hst_ptrlcl_q_mem_size;
  size_t hst_ptrlcl_q_inv_mem_size;
  size_t hst_ptrlevel_mem_size;
  size_t hst_ptrlevel_int_mem_size;
  size_t hst_ptrresult_mem_size;

    size_t hst_ptralpha_dim1;
  size_t hst_ptrindex_dim1;
  size_t hst_ptrindex_dim2;
  size_t hst_ptrlambda_dim1;
  size_t hst_ptrlcl_q_dim1;
  size_t hst_ptrlcl_q_inv_dim1;
  size_t hst_ptrlevel_dim1;
  size_t hst_ptrlevel_dim2;
  size_t hst_ptrlevel_int_dim1;
  size_t hst_ptrlevel_int_dim2;
  size_t hst_ptrresult_dim1;

    size_t isFirstTime = 1;
  std::string KernelDefines = "";
  Stopwatch timer;


public:
  OCLLaplaceTask()
  {
    isFirstTime = 1;
    KernelDefines = 1;
  }

  void RunOCLLaplaceForKernel(
	double * arg_alpha, size_t arg_hst_ptralpha_dim1, size_t arg_dim, 
	double * arg_index, size_t arg_hst_ptrindex_dim1, size_t arg_hst_ptrindex_dim2, 
	double * arg_lambda, size_t arg_hst_ptrlambda_dim1, double * arg_lcl_q, 
	size_t arg_hst_ptrlcl_q_dim1, double * arg_lcl_q_inv, size_t arg_hst_ptrlcl_q_inv_dim1, 
	double * arg_level, size_t arg_hst_ptrlevel_dim1, size_t arg_hst_ptrlevel_dim2, 
	double * arg_level_int, size_t arg_hst_ptrlevel_int_dim1, size_t arg_hst_ptrlevel_int_dim2, 
	std::string arg_ocl_type, double * arg_result, size_t arg_hst_ptrresult_dim1, 
	size_t arg_storagesize)
  {
    if (isFirstTime)
      {
        hst_ptralpha = arg_alpha;
        hst_ptralpha_dim1 = arg_hst_ptralpha_dim1;
        dim = arg_dim;
        hst_ptrindex = arg_index;
        hst_ptrindex_dim1 = arg_hst_ptrindex_dim1;
        hst_ptrindex_dim2 = arg_hst_ptrindex_dim2;
        hst_ptrlambda = arg_lambda;
        hst_ptrlambda_dim1 = arg_hst_ptrlambda_dim1;
        hst_ptrlcl_q = arg_lcl_q;
        hst_ptrlcl_q_dim1 = arg_hst_ptrlcl_q_dim1;
        hst_ptrlcl_q_inv = arg_lcl_q_inv;
        hst_ptrlcl_q_inv_dim1 = arg_hst_ptrlcl_q_inv_dim1;
        hst_ptrlevel = arg_level;
        hst_ptrlevel_dim1 = arg_hst_ptrlevel_dim1;
        hst_ptrlevel_dim2 = arg_hst_ptrlevel_dim2;
        hst_ptrlevel_int = arg_level_int;
        hst_ptrlevel_int_dim1 = arg_hst_ptrlevel_int_dim1;
        hst_ptrlevel_int_dim2 = arg_hst_ptrlevel_int_dim2;
        ocl_type = arg_ocl_type;
        hst_ptrresult = arg_result;
        hst_ptrresult_dim1 = arg_hst_ptrresult_dim1;
        storagesize = arg_storagesize;
        StartUpOCL(ocl_type);
        AllocateBuffers();
        cout << "$Defines " << KernelDefines << endl;
        compileKernel(
	"LaplaceFor", "LaplaceFor.cl", GetKernelCode(), 
	false, &LaplaceForKernel, KernelDefines
	);
        SetArgumentsLaplaceFor();
      }
    timer.start();
    ExecLaplaceFor();
    cout << "$Time " << timer.stop() << endl;
  }


private:
  std::string LaplaceBase()
  {
    std::stringstream str;
    str << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    str << "#include \"LaplaceIncludes.hpp\"" << endl;
    str << "__kernel void LaplaceFor(" << endl;
    str << "	__global double * alpha, __global double * index, __global double * lambda, " << endl;
    str << "	__global double * lcl_q, __global double * lcl_q_inv, __global double * level, " << endl;
    str << "	__global double * level_int, __global double * result) {" << endl;
    str << "  for (unsigned j = 0; j < storagesize; j++) {" << endl;
    str << "      double gradient_temp[dim];" << endl;
    str << "      double dot_temp[dim];" << endl;
    str << "      for (unsigned d = 0; d < dim; d++) {" << endl;
    str << "          double level_i = level[(d * hst_ptrlevel_dim1) + get_global_id(0)];" << endl;
    str << "          double level_j = level[(d * hst_ptrlevel_dim1) + j];" << endl;
    str << "          double level_int_i = level_int[(d * hst_ptrlevel_int_dim1) + get_global_id(0)];" << endl;
    str << "          double level_int_j = level_int[(d * hst_ptrlevel_int_dim1) + j];" << endl;
    str << "          double index_i = index[(d * hst_ptrindex_dim1) + get_global_id(0)];" << endl;
    str << "          double index_j = index[(d * hst_ptrindex_dim1) + j];" << endl;
    str << "          gradient_temp[d] = gradient(" << endl;
    str << "	level_i, index_i, level_j, " << endl;
    str << "	index_j, lcl_q_inv[d]);" << endl;
    str << "          dot_temp[d] = l2dot(" << endl;
    str << "	level_i, level_j, index_i, " << endl;
    str << "	index_j, level_int_i, level_int_j, " << endl;
    str << "	lcl_q[d]);" << endl;
    str << "      }" << endl;
    str << "      double sub = 0.0;" << endl;
    str << "      for (unsigned d_outer = 0; d_outer < dim; d_outer++) {" << endl;
    str << "          double element = alpha[j];" << endl;
    str << "          for (unsigned d_inner = 0; d_inner < dim; d_inner++) {" << endl;
    str << "              element *= (dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner));" << endl;
    str << "          }" << endl;
    str << "          sub += lambda[d_outer] * element;" << endl;
    str << "      }" << endl;
    str << "      result[get_global_id(0)] += sub;" << endl;
    str << "  }" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string LaplacePlaceInReg()
  {
    std::stringstream str;
    str << "#pragma OPENCL EXTENSION cl_khr_fp64: enable" << endl;
    str << "#include \"LaplaceIncludes.hpp\"" << endl;
    str << "__kernel void LaplaceFor(" << endl;
    str << "	__global double * alpha, __global double * index, __global double * lambda, " << endl;
    str << "	__global double * lcl_q, __global double * lcl_q_inv, __global double * level, " << endl;
    str << "	__global double * level_int, __global double * result) {" << endl;
    str << "  double index_reg[dim];" << endl;
    str << "  double level_int_reg[dim];" << endl;
    str << "  double level_reg[dim];" << endl;
    str << "  for (unsigned d = 0; d < dim; d++) {" << endl;
    str << "      index_reg[d] = index[(d * hst_ptrindex_dim1) + get_global_id(0)];" << endl;
    str << "      level_int_reg[d] = level_int[(d * hst_ptrlevel_int_dim1) + get_global_id(0)];" << endl;
    str << "      level_reg[d] = level[(d * hst_ptrlevel_dim1) + get_global_id(0)];" << endl;
    str << "  }" << endl;
    str << "  for (unsigned j = 0; j < storagesize; j++) {" << endl;
    str << "      double gradient_temp[dim];" << endl;
    str << "      double dot_temp[dim];" << endl;
    str << "      for (unsigned d = 0; d < dim; d++) {" << endl;
    str << "          double level_i = level_reg[d];" << endl;
    str << "          double level_j = level[(d * hst_ptrlevel_dim1) + j];" << endl;
    str << "          double level_int_i = level_int_reg[d];" << endl;
    str << "          double level_int_j = level_int[(d * hst_ptrlevel_int_dim1) + j];" << endl;
    str << "          double index_i = index_reg[d];" << endl;
    str << "          double index_j = index[(d * hst_ptrindex_dim1) + j];" << endl;
    str << "          gradient_temp[d] = gradient(" << endl;
    str << "	level_i, index_i, level_j, " << endl;
    str << "	index_j, lcl_q_inv[d]);" << endl;
    str << "          dot_temp[d] = l2dot(" << endl;
    str << "	level_i, level_j, index_i, " << endl;
    str << "	index_j, level_int_i, level_int_j, " << endl;
    str << "	lcl_q[d]);" << endl;
    str << "      }" << endl;
    str << "      double sub = 0.0;" << endl;
    str << "      for (unsigned d_outer = 0; d_outer < dim; d_outer++) {" << endl;
    str << "          double element = alpha[j];" << endl;
    str << "          for (unsigned d_inner = 0; d_inner < dim; d_inner++) {" << endl;
    str << "              element *= (dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner));" << endl;
    str << "          }" << endl;
    str << "          sub += lambda[d_outer] * element;" << endl;
    str << "      }" << endl;
    str << "      result[get_global_id(0)] += sub;" << endl;
    str << "  }" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string GetKernelCode()
  {
    if (((dim - 0) * 3) < 40)
      {
        return LaplacePlaceInReg();
      }
    else
      {
        return LaplaceBase();
      }
  }

  void AllocateBuffers()
  {
        hst_ptralpha_mem_size = hst_ptralpha_dim1 * sizeof(double);
    hst_ptrindex_mem_size = hst_ptrindex_dim2 * (hst_ptrindex_dim1 * sizeof(double));
    hst_ptrlambda_mem_size = hst_ptrlambda_dim1 * sizeof(double);
    hst_ptrlcl_q_mem_size = hst_ptrlcl_q_dim1 * sizeof(double);
    hst_ptrlcl_q_inv_mem_size = hst_ptrlcl_q_inv_dim1 * sizeof(double);
    hst_ptrlevel_mem_size = hst_ptrlevel_dim2 * (hst_ptrlevel_dim1 * sizeof(double));
    hst_ptrlevel_int_mem_size = hst_ptrlevel_int_dim2 * (hst_ptrlevel_int_dim1 * sizeof(double));
    hst_ptrresult_mem_size = hst_ptrresult_dim1 * sizeof(double);

        // Transposition
    hst_ptrindex_trans = new double[hst_ptrindex_mem_size];
    transpose<double>(
	hst_ptrindex, hst_ptrindex_trans, hst_ptrindex_dim1, 
	hst_ptrindex_dim2);
    hst_ptrlevel_int_trans = new double[hst_ptrlevel_int_mem_size];
    transpose<double>(
	hst_ptrlevel_int, hst_ptrlevel_int_trans, hst_ptrlevel_int_dim1, 
	hst_ptrlevel_int_dim2);
    hst_ptrlevel_trans = new double[hst_ptrlevel_mem_size];
    transpose<double>(
	hst_ptrlevel, hst_ptrlevel_trans, hst_ptrlevel_dim1, 
	hst_ptrlevel_dim2);

        // Constant Memory

        // Defines for the kernel
    std::stringstream str;
    str << "-Ddim=" << dim << " ";
    str << "-Dhst_ptrindex_dim1=" << hst_ptrindex_dim2 << " ";
    str << "-Dhst_ptrlevel_dim1=" << hst_ptrlevel_dim2 << " ";
    str << "-Dhst_ptrlevel_int_dim1=" << hst_ptrlevel_int_dim2 << " ";
    str << "-Dstoragesize=" << storagesize << " ";
    KernelDefines = str.str();

        cl_int oclErrNum = CL_SUCCESS;

    dev_ptralpha = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptralpha_mem_size, 
	hst_ptralpha, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptralpha");
    dev_ptrindex = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrindex_mem_size, 
	hst_ptrindex_trans, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrindex");
    dev_ptrlambda = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlambda_mem_size, 
	hst_ptrlambda, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlambda");
    dev_ptrlcl_q = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlcl_q_mem_size, 
	hst_ptrlcl_q, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlcl_q");
    dev_ptrlcl_q_inv = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlcl_q_inv_mem_size, 
	hst_ptrlcl_q_inv, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlcl_q_inv");
    dev_ptrlevel = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlevel_mem_size, 
	hst_ptrlevel_trans, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlevel");
    dev_ptrlevel_int = clCreateBuffer(
	context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrlevel_int_mem_size, 
	hst_ptrlevel_int_trans, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrlevel_int");
    dev_ptrresult = clCreateBuffer(
	context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrresult_mem_size, 
	hst_ptrresult, &oclErrNum);
    oclCheckErr(
	oclErrNum, "clCreateBuffer dev_ptrresult");
  }

  void SetArgumentsLaplaceFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    int counter = 0;
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptralpha);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrindex);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlambda);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlcl_q);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlcl_q_inv);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlevel);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrlevel_int);
    oclErrNum |= clSetKernelArg(
	LaplaceForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrresult);
    oclCheckErr(
	oclErrNum, "clSetKernelArg");
  }

  void ExecLaplaceFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    cl_event GPUExecution;
    size_t LaplaceFor_global_worksize[] = {storagesize - 0};
    size_t LaplaceFor_local_worksize[] = {16};
    size_t LaplaceFor_global_offset[] = {0};
    oclErrNum = clEnqueueNDRangeKernel(
	command_queue, LaplaceForKernel, 1, 
	LaplaceFor_global_offset, LaplaceFor_global_worksize, LaplaceFor_local_worksize, 
	0, NULL, &GPUExecution
	);
    oclCheckErr(
	oclErrNum, "clEnqueueNDRangeKernel");
    oclErrNum = clFinish(command_queue);
    oclCheckErr(
	oclErrNum, "clFinish");
    oclErrNum = clEnqueueReadBuffer(
	command_queue, dev_ptrresult, CL_TRUE, 
	0, hst_ptrresult_mem_size, hst_ptrresult, 
	1, &GPUExecution, NULL
	);
    oclCheckErr(
	oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clFinish(command_queue);
    oclCheckErr(
	oclErrNum, "clFinish");
  }


}
