#include "StartUtil.cpp"
#include "helper.hpp"
using namespace std;

class OCLGaussianDerivatesTask
{
  cl_kernel GaussianDerivatesForKernel;
  cl_mem dev_ptrD1Ks__ijb_dimsI;
  cl_mem dev_ptrD1Ks__ijb_x;
  cl_mem dev_ptrD2Ks__ijbg_dimsI;
  cl_mem dev_ptrD2Ks__ijbg_x;
  cl_mem dev_ptrD3Ks__ijbgd_dimsI;
  cl_mem dev_ptrD3Ks__ijbgd_x;
  cl_mem dev_ptrK__ij_x;
  cl_mem dev_ptrp_a_i_x;
  cl_mem dev_ptrq_a_i_x;

  unsigned * hst_ptrD1Ks__ijb_dimsI;
  float * hst_ptrD1Ks__ijb_x;
  unsigned * hst_ptrD2Ks__ijbg_dimsI;
  float * hst_ptrD2Ks__ijbg_x;
  unsigned * hst_ptrD3Ks__ijbgd_dimsI;
  float * hst_ptrD3Ks__ijbgd_x;
  unsigned dim;
  float * hst_ptrK__ij_x;
  unsigned Lp;
  unsigned Lq;
  std::string ocl_type;
  float * hst_ptrp_a_i_x;
  float * hst_ptrq_a_i_x;
  float scales2_x;
  float scaleweight2_x;
  float * hst_ptrq_a_i_x_trans;

  size_t hst_ptrD1Ks__ijb_dimsI_mem_size;
  size_t hst_ptrD1Ks__ijb_x_mem_size;
  size_t hst_ptrD2Ks__ijbg_dimsI_mem_size;
  size_t hst_ptrD2Ks__ijbg_x_mem_size;
  size_t hst_ptrD3Ks__ijbgd_dimsI_mem_size;
  size_t hst_ptrD3Ks__ijbgd_x_mem_size;
  size_t hst_ptrK__ij_x_mem_size;
  size_t hst_ptrp_a_i_x_mem_size;
  size_t hst_ptrq_a_i_x_mem_size;

  size_t hst_ptrD1Ks__ijb_dimsI_dim1;
  size_t hst_ptrD1Ks__ijb_x_dim1;
  size_t hst_ptrD2Ks__ijbg_dimsI_dim1;
  size_t hst_ptrD2Ks__ijbg_x_dim1;
  size_t hst_ptrD3Ks__ijbgd_dimsI_dim1;
  size_t hst_ptrD3Ks__ijbgd_x_dim1;
  size_t hst_ptrK__ij_x_dim1;
  size_t hst_ptrK__ij_x_dim2;
  size_t hst_ptrp_a_i_x_dim1;
  size_t hst_ptrp_a_i_x_dim2;
  size_t hst_ptrq_a_i_x_dim1;
  size_t hst_ptrq_a_i_x_dim2;

  size_t isFirstTime;
  std::string KernelDefines;
  Stopwatch timer;
  OCLContext * ocl_context;


public:
  OCLGaussianDerivatesTask()
  {
    isFirstTime = 1;
    KernelDefines = "";
  }

  void RunOCLGaussianDerivatesForKernel(unsigned * arg_D1Ks__ijb_dimsI, size_t arg_hst_ptrD1Ks__ijb_dimsI_dim1, float * arg_D1Ks__ijb_x, 
	size_t arg_hst_ptrD1Ks__ijb_x_dim1, unsigned * arg_D2Ks__ijbg_dimsI, size_t arg_hst_ptrD2Ks__ijbg_dimsI_dim1, 
	float * arg_D2Ks__ijbg_x, size_t arg_hst_ptrD2Ks__ijbg_x_dim1, unsigned * arg_D3Ks__ijbgd_dimsI, 
	size_t arg_hst_ptrD3Ks__ijbgd_dimsI_dim1, float * arg_D3Ks__ijbgd_x, size_t arg_hst_ptrD3Ks__ijbgd_x_dim1, 
	float * arg_K__ij_x, size_t arg_hst_ptrK__ij_x_dim1, size_t arg_hst_ptrK__ij_x_dim2, 
	unsigned arg_Lp, unsigned arg_Lq, unsigned arg_dim, 
	std::string arg_ocl_type, float * arg_p_a_i_x, size_t arg_hst_ptrp_a_i_x_dim1, 
	size_t arg_hst_ptrp_a_i_x_dim2, float * arg_q_a_i_x, size_t arg_hst_ptrq_a_i_x_dim1, 
	size_t arg_hst_ptrq_a_i_x_dim2, float arg_scales2_x, float arg_scaleweight2_x
	)
  {
    if (isFirstTime)
    {
      hst_ptrD1Ks__ijb_dimsI = arg_D1Ks__ijb_dimsI;
      hst_ptrD1Ks__ijb_dimsI_dim1 = arg_hst_ptrD1Ks__ijb_dimsI_dim1;
      hst_ptrD1Ks__ijb_x = arg_D1Ks__ijb_x;
      hst_ptrD1Ks__ijb_x_dim1 = arg_hst_ptrD1Ks__ijb_x_dim1;
      hst_ptrD2Ks__ijbg_dimsI = arg_D2Ks__ijbg_dimsI;
      hst_ptrD2Ks__ijbg_dimsI_dim1 = arg_hst_ptrD2Ks__ijbg_dimsI_dim1;
      hst_ptrD2Ks__ijbg_x = arg_D2Ks__ijbg_x;
      hst_ptrD2Ks__ijbg_x_dim1 = arg_hst_ptrD2Ks__ijbg_x_dim1;
      hst_ptrD3Ks__ijbgd_dimsI = arg_D3Ks__ijbgd_dimsI;
      hst_ptrD3Ks__ijbgd_dimsI_dim1 = arg_hst_ptrD3Ks__ijbgd_dimsI_dim1;
      hst_ptrD3Ks__ijbgd_x = arg_D3Ks__ijbgd_x;
      hst_ptrD3Ks__ijbgd_x_dim1 = arg_hst_ptrD3Ks__ijbgd_x_dim1;
      hst_ptrK__ij_x = arg_K__ij_x;
      hst_ptrK__ij_x_dim1 = arg_hst_ptrK__ij_x_dim1;
      hst_ptrK__ij_x_dim2 = arg_hst_ptrK__ij_x_dim2;
      Lp = arg_Lp;
      Lq = arg_Lq;
      dim = arg_dim;
      ocl_type = arg_ocl_type;
      hst_ptrp_a_i_x = arg_p_a_i_x;
      hst_ptrp_a_i_x_dim1 = arg_hst_ptrp_a_i_x_dim1;
      hst_ptrp_a_i_x_dim2 = arg_hst_ptrp_a_i_x_dim2;
      hst_ptrq_a_i_x = arg_q_a_i_x;
      hst_ptrq_a_i_x_dim1 = arg_hst_ptrq_a_i_x_dim1;
      hst_ptrq_a_i_x_dim2 = arg_hst_ptrq_a_i_x_dim2;
      scales2_x = arg_scales2_x;
      scaleweight2_x = arg_scaleweight2_x;
      ocl_context = new OCLContext();
      ocl_context->StartUpOCL(ocl_type);
      AllocateBuffers();
      ocl_context->compileKernel("GaussianDerivatesFor",  GetKernelCode(), 
	 &GaussianDerivatesForKernel, KernelDefines
	);
      SetArgumentsGaussianDerivatesFor();
    }
    timer.start();
    ExecGaussianDerivatesFor();
    cout << "$Sequential_time " << timer.stop() << endl;
  }


private:
  std::string GaussianDerivatesBase()
  {
    std::stringstream str;
    str << "#include \"GaussianDerivatesIncludes.hpp\"" << endl;
    str << "__kernel void GaussianDerivatesFor(" << endl;
    str << "  __global unsigned * D1Ks__ijb_dimsI, __global float * D1Ks__ijb_x, __global unsigned * D2Ks__ijbg_dimsI, " << endl;
    str << "  __global float * D2Ks__ijbg_x, __global unsigned * D3Ks__ijbgd_dimsI, __global float * D3Ks__ijbgd_x, " << endl;
    str << "  __global float * K__ij_x, __global float * p_a_i_x, __global float * q_a_i_x" << endl;
    str << "  ) {" << endl;
    str << "  float xj[3];" << endl;
    str << "  float xi[3];" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      xj[k] = p_a_i_x[(get_global_id(1) * hst_ptrp_a_i_x_dim1) + k];" << endl;
    str << "  }" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      xi[k] = q_a_i_x[(k * hst_ptrq_a_i_x_dim1) + get_global_id(0)];" << endl;
    str << "  }" << endl;
    str << "  // Vector3<scalar> xi(&q_a_i.x[q_a_i.rows*i]);" << endl;
    str << "  float ximxj[3];" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      ximxj[k] = xi[k] - xj[k];" << endl;
    str << "  }" << endl;
    str << "  float r = sqrt(scales2_x);" << endl;
    str << "  float ks = gamma(" << endl;
    str << "  ximxj, scales2_x, scaleweight2_x" << endl;
    str << "  );" << endl;
    str << "  K__ij_x[(get_global_id(1) * hst_ptrK__ij_x_dim1) + get_global_id(0)] = ks;" << endl;
    str << "  int da[3];" << endl;
    str << "  int db[3];" << endl;
    str << "  int dc[3];" << endl;
    str << "  // nargout 1" << endl;
    str << "  for (int b = 0; b < dim; b++) {" << endl;
    str << "      // da.set(1,b);" << endl;
    str << "      for (int k = 0; k < dim; k++) {" << endl;
    str << "          da[k] = 1;" << endl;
    str << "      }" << endl;
    str << "      D1Ks__ijb_x[(get_global_id(0) + (D1Ks__ijb_dimsI[0] * get_global_id(1))) + (D1Ks__ijb_dimsI[1] * b)] = DaKs(" << endl;
    str << "  da, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "      // nargout 2" << endl;
    str << "      for (int g = 0; g < dim; g++) {" << endl;
    str << "          // Vector3<int> db = da;" << endl;
    str << "          for (int k = 0; k < dim; k++) {" << endl;
    str << "              db[k] = da[k] + 1;" << endl;
    str << "          }" << endl;
    str << "          // db.set(db[g]+1,g) ?" << endl;
    str << "          // db[g] = db[g] + 1;" << endl;
    str << "          // for (int k = 0; k < dim; k++) {" << endl;
    str << "          //   db[g] = db[g] + 1;" << endl;
    str << "          // }" << endl;
    str << "          D2Ks__ijbg_x[((get_global_id(0) + (D2Ks__ijbg_dimsI[0] * get_global_id(1))) + (D2Ks__ijbg_dimsI[1] * b)) + (D2Ks__ijbg_dimsI[2] * g)] = DaKs(" << endl;
    str << "  db, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "          for (int d = 0; d < dim; d++) {" << endl;
    str << "              // Vector3<int> dc = db; dc.set(dc[d]+1,d);" << endl;
    str << "              for (int k = 0; k < dim; k++) {" << endl;
    str << "                  dc[k] = db[k] + 1;" << endl;
    str << "              }" << endl;
    str << "              // for (int k = 0; k < dim; k++) {" << endl;
    str << "              //   dc[d] = dc[d] + 1;" << endl;
    str << "              // }	    " << endl;
    str << "              D3Ks__ijbgd_x[(((get_global_id(0) + (D3Ks__ijbgd_dimsI[0] * get_global_id(1))) + (D3Ks__ijbgd_dimsI[1] * b)) + (D3Ks__ijbgd_dimsI[2] * g)) + (D3Ks__ijbgd_dimsI[3] * d)] = DaKs(" << endl;
    str << "  dc, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "          }" << endl;
    str << "      }" << endl;
    str << "  }" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string GaussianDerivatesPlaceInLocal()
  {
    std::stringstream str;
    str << "#include \"GaussianDerivatesIncludes.hpp\"" << endl;
    str << "__kernel void GaussianDerivatesFor(" << endl;
    str << "  __global unsigned * D1Ks__ijb_dimsI, __global float * D1Ks__ijb_x, __global unsigned * D2Ks__ijbg_dimsI, " << endl;
    str << "  __global float * D2Ks__ijbg_x, __global unsigned * D3Ks__ijbgd_dimsI, __global float * D3Ks__ijbgd_x, " << endl;
    str << "  __global float * K__ij_x, __global float * p_a_i_x, __global float * q_a_i_x" << endl;
    str << "  ) {" << endl;
    str << "  __local float p_a_i_x_local[4 * 4];" << endl;
    str << "  __local float q_a_i_x_local[4 * 4];" << endl;
    str << "  float xj[3];" << endl;
    str << "  float xi[3];" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      xj[k] = p_a_i_x_local[(get_local_id(1) * 4) + kk];" << endl;
    str << "  }" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      xi[k] = q_a_i_x_local[(kk * 4) + get_local_id(0)];" << endl;
    str << "  }" << endl;
    str << "  // Vector3<scalar> xi(&q_a_i.x[q_a_i.rows*i]);" << endl;
    str << "  float ximxj[3];" << endl;
    str << "  for (int k = 0; k < dim; k++) {" << endl;
    str << "      ximxj[k] = xi[k] - xj[k];" << endl;
    str << "  }" << endl;
    str << "  float r = sqrt(scales2_x);" << endl;
    str << "  float ks = gamma(" << endl;
    str << "  ximxj, scales2_x, scaleweight2_x" << endl;
    str << "  );" << endl;
    str << "  K__ij_x[(get_global_id(1) * hst_ptrK__ij_x_dim1) + get_global_id(0)] = ks;" << endl;
    str << "  int da[3];" << endl;
    str << "  int db[3];" << endl;
    str << "  int dc[3];" << endl;
    str << "  // nargout 1" << endl;
    str << "  for (int b = 0; b < dim; b++) {" << endl;
    str << "      // da.set(1,b);" << endl;
    str << "      for (int k = 0; k < dim; k++) {" << endl;
    str << "          da[k] = 1;" << endl;
    str << "      }" << endl;
    str << "      D1Ks__ijb_x[(get_global_id(0) + (D1Ks__ijb_dimsI[0] * get_global_id(1))) + (D1Ks__ijb_dimsI[1] * b)] = DaKs(" << endl;
    str << "  da, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "      // nargout 2" << endl;
    str << "      for (int g = 0; g < dim; g++) {" << endl;
    str << "          // Vector3<int> db = da;" << endl;
    str << "          for (int k = 0; k < dim; k++) {" << endl;
    str << "              db[k] = da[k] + 1;" << endl;
    str << "          }" << endl;
    str << "          // db.set(db[g]+1,g) ?" << endl;
    str << "          // db[g] = db[g] + 1;" << endl;
    str << "          // for (int k = 0; k < dim; k++) {" << endl;
    str << "          //   db[g] = db[g] + 1;" << endl;
    str << "          // }" << endl;
    str << "          D2Ks__ijbg_x[((get_global_id(0) + (D2Ks__ijbg_dimsI[0] * get_global_id(1))) + (D2Ks__ijbg_dimsI[1] * b)) + (D2Ks__ijbg_dimsI[2] * g)] = DaKs(" << endl;
    str << "  db, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "          for (int d = 0; d < dim; d++) {" << endl;
    str << "              // Vector3<int> dc = db; dc.set(dc[d]+1,d);" << endl;
    str << "              for (int k = 0; k < dim; k+=4) {" << endl;
    str << "                  p_a_i_x_local[(get_local_id(1) * 4) + get_local_id(0)] = p_a_i_x[(get_global_id(1) * hst_ptrp_a_i_x_dim1) + (k + get_local_id(0))];" << endl;
    str << "                  q_a_i_x_local[(get_local_id(1) * 4) + get_local_id(0)] = q_a_i_x[((k + get_local_id(1)) * hst_ptrq_a_i_x_dim1) + get_global_id(0)];" << endl;
    str << "                  barrier(CLK_LOCAL_MEM_FENCE);" << endl;
    str << "                  for (unsigned kk = 0; kk < 4; kk++) {" << endl;
    str << "                      dc[k] = db[k] + 1;" << endl;
    str << "                  }" << endl;
    str << "                  barrier(CLK_LOCAL_MEM_FENCE);" << endl;
    str << "              }" << endl;
    str << "              // for (int k = 0; k < dim; k++) {" << endl;
    str << "              //   dc[d] = dc[d] + 1;" << endl;
    str << "              // }	    " << endl;
    str << "              D3Ks__ijbgd_x[(((get_global_id(0) + (D3Ks__ijbgd_dimsI[0] * get_global_id(1))) + (D3Ks__ijbgd_dimsI[1] * b)) + (D3Ks__ijbgd_dimsI[2] * g)) + (D3Ks__ijbgd_dimsI[3] * d)] = DaKs(" << endl;
    str << "  dc, ximxj, r, " << endl;
    str << "  ks);" << endl;
    str << "          }" << endl;
    str << "      }" << endl;
    str << "  }" << endl;
    str << "}" << endl;
    
    return str.str();
  }


  std::string GetKernelCode()
  {
    if (((dim - 0) % 4) == 0)
    {
      return GaussianDerivatesPlaceInLocal();
    }    
    else
    {
      return GaussianDerivatesBase();
    }
  }

  void AllocateBuffers()
  {
    hst_ptrD1Ks__ijb_dimsI_mem_size = hst_ptrD1Ks__ijb_dimsI_dim1 * sizeof(unsigned);
    hst_ptrD1Ks__ijb_x_mem_size = hst_ptrD1Ks__ijb_x_dim1 * sizeof(float);
    hst_ptrD2Ks__ijbg_dimsI_mem_size = hst_ptrD2Ks__ijbg_dimsI_dim1 * sizeof(unsigned);
    hst_ptrD2Ks__ijbg_x_mem_size = hst_ptrD2Ks__ijbg_x_dim1 * sizeof(float);
    hst_ptrD3Ks__ijbgd_dimsI_mem_size = hst_ptrD3Ks__ijbgd_dimsI_dim1 * sizeof(unsigned);
    hst_ptrD3Ks__ijbgd_x_mem_size = hst_ptrD3Ks__ijbgd_x_dim1 * sizeof(float);
    hst_ptrK__ij_x_mem_size = hst_ptrK__ij_x_dim2 * (hst_ptrK__ij_x_dim1 * sizeof(float));
    hst_ptrp_a_i_x_mem_size = hst_ptrp_a_i_x_dim2 * (hst_ptrp_a_i_x_dim1 * sizeof(float));
    hst_ptrq_a_i_x_mem_size = hst_ptrq_a_i_x_dim2 * (hst_ptrq_a_i_x_dim1 * sizeof(float));

    // Transposition
    hst_ptrq_a_i_x_trans = new float[hst_ptrq_a_i_x_mem_size];
    helper::transpose<float>(hst_ptrq_a_i_x, hst_ptrq_a_i_x_trans, hst_ptrq_a_i_x_dim1, 
	hst_ptrq_a_i_x_dim2);

    // Constant Memory

    // Defines for the kernel
    std::stringstream str;
    str << "-Ddim=" << dim << " ";
    str << "-Dhst_ptrK__ij_x_dim1=" << hst_ptrK__ij_x_dim1 << " ";
    str << "-Dhst_ptrp_a_i_x_dim1=" << hst_ptrp_a_i_x_dim1 << " ";
    str << "-Dhst_ptrq_a_i_x_dim1=" << hst_ptrq_a_i_x_dim2 << " ";
    str << "-Dscales2_x=" << scales2_x << " ";
    str << "-Dscaleweight2_x=" << scaleweight2_x << " ";
    KernelDefines = str.str();

    cl_int oclErrNum = CL_SUCCESS;

    dev_ptrD1Ks__ijb_dimsI = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrD1Ks__ijb_dimsI_mem_size, 
	hst_ptrD1Ks__ijb_dimsI, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD1Ks__ijb_dimsI");
    dev_ptrD1Ks__ijb_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrD1Ks__ijb_x_mem_size, 
	hst_ptrD1Ks__ijb_x, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD1Ks__ijb_x");
    dev_ptrD2Ks__ijbg_dimsI = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrD2Ks__ijbg_dimsI_mem_size, 
	hst_ptrD2Ks__ijbg_dimsI, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD2Ks__ijbg_dimsI");
    dev_ptrD2Ks__ijbg_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrD2Ks__ijbg_x_mem_size, 
	hst_ptrD2Ks__ijbg_x, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD2Ks__ijbg_x");
    dev_ptrD3Ks__ijbgd_dimsI = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrD3Ks__ijbgd_dimsI_mem_size, 
	hst_ptrD3Ks__ijbgd_dimsI, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD3Ks__ijbgd_dimsI");
    dev_ptrD3Ks__ijbgd_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrD3Ks__ijbgd_x_mem_size, 
	hst_ptrD3Ks__ijbgd_x, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrD3Ks__ijbgd_x");
    dev_ptrK__ij_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, hst_ptrK__ij_x_mem_size, 
	hst_ptrK__ij_x, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrK__ij_x");
    dev_ptrp_a_i_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrp_a_i_x_mem_size, 
	hst_ptrp_a_i_x, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrp_a_i_x");
    dev_ptrq_a_i_x = clCreateBuffer(ocl_context->getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hst_ptrq_a_i_x_mem_size, 
	hst_ptrq_a_i_x_trans, &oclErrNum);
    helper::oclCheckErr(oclErrNum, "clCreateBuffer dev_ptrq_a_i_x");
  }

  void SetArgumentsGaussianDerivatesFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    int counter = 0;
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD1Ks__ijb_dimsI);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD1Ks__ijb_x);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD2Ks__ijbg_dimsI);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD2Ks__ijbg_x);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD3Ks__ijbgd_dimsI);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrD3Ks__ijbgd_x);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrK__ij_x);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrp_a_i_x);
    oclErrNum |= clSetKernelArg(GaussianDerivatesForKernel, counter++, sizeof(cl_mem), 
	(void *) &dev_ptrq_a_i_x);
    helper::oclCheckErr(oclErrNum, "clSetKernelArg");
  }

  void ExecGaussianDerivatesFor()
  {
    cl_int oclErrNum = CL_SUCCESS;
    cl_event GPUExecution;
    size_t GaussianDerivatesFor_global_worksize[] = {Lq - 0, Lp - 0};
    size_t GaussianDerivatesFor_local_worksize[] = {4, 4};
    size_t GaussianDerivatesFor_global_offset[] = {0, 0};
    oclErrNum = clEnqueueNDRangeKernel(ocl_context->getCommandQueue(), GaussianDerivatesForKernel, 2, 
	GaussianDerivatesFor_global_offset, GaussianDerivatesFor_global_worksize, GaussianDerivatesFor_local_worksize, 
	0, NULL, &GPUExecution
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueNDRangeKernel");
    oclErrNum = clFinish(ocl_context->getCommandQueue());
    helper::oclCheckErr(oclErrNum, "clFinish");
    oclErrNum = clEnqueueReadBuffer(ocl_context->getCommandQueue(), dev_ptrD1Ks__ijb_x, CL_TRUE, 
	0, hst_ptrD1Ks__ijb_x_mem_size, hst_ptrD1Ks__ijb_x, 
	1, &GPUExecution, NULL
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clEnqueueReadBuffer(ocl_context->getCommandQueue(), dev_ptrD2Ks__ijbg_x, CL_TRUE, 
	0, hst_ptrD2Ks__ijbg_x_mem_size, hst_ptrD2Ks__ijbg_x, 
	1, &GPUExecution, NULL
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clEnqueueReadBuffer(ocl_context->getCommandQueue(), dev_ptrD3Ks__ijbgd_x, CL_TRUE, 
	0, hst_ptrD3Ks__ijbgd_x_mem_size, hst_ptrD3Ks__ijbgd_x, 
	1, &GPUExecution, NULL
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clEnqueueReadBuffer(ocl_context->getCommandQueue(), dev_ptrK__ij_x, CL_TRUE, 
	0, hst_ptrK__ij_x_mem_size, hst_ptrK__ij_x, 
	1, &GPUExecution, NULL
	);
    helper::oclCheckErr(oclErrNum, "clEnqueueReadBuffer");
    oclErrNum = clFinish(ocl_context->getCommandQueue());
    helper::oclCheckErr(oclErrNum, "clFinish");
  }


};
