#include "OCLLaplaceInner.hpp"

namespace sg {
  namespace parallel {
    namespace oclpdekernels {

      cl_kernel GenAInnerKernel[NUMDEVS];
      cl_mem d_ptrAInner[NUMDEVS];
      REAL * ptrAInner;
      size_t A_max_size;

      void SetArgumentsGenAInner() {
	cl_int ciErrNum = CL_SUCCESS;
	int counter = 0;

	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelInner[i]);
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconInner[i]);

	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexInner[i]);
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intInner[i]);

	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaInner[i]);
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAInner[i]);
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qInner[i]);
	    ciErrNum |= clSetKernelArg(GenAInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLambdaInner[i]);
	    oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");

	    counter = 0;
	  }
      }

      void SetABufferGenAInner() {
	cl_int ciErrNum = CL_SUCCESS;
  

	size_t level_size = storageSizePadded * dims;
	size_t index_size = storageSizePadded * dims;
	size_t level_int_size = storageSizePadded * dims;
	size_t alpha_size = storageSizePadded;
	size_t result_size = storageSizePadded; 
	size_t A_size = storageSizePadded*storageSizePadded;

	ptrAInner = (double*)calloc(A_size,sizeof(REAL));


	cl_ulong size3;
	clGetDeviceInfo(device_ids[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size3, 0);

	cl_ulong size4;
	clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), &size4, 0);

	size_t sizedoubles64 = size3 / sizeof(REAL);
	size_t gpu_max_buffer_size = size4 / sizeof(REAL);
	gpu_max_buffer_size = gpu_max_buffer_size - (gpu_max_buffer_size % (storageSizePadded*LSIZE));
	
	size_t memoryNonParResult = level_size + index_size + level_int_size + alpha_size + result_size;
	size_t memoryLeftover = sizedoubles64 - memoryNonParResult;

	A_max_size = memoryLeftover - (memoryLeftover % (storageSizePadded*LSIZE));
	A_max_size = std::min(A_max_size, A_size);
	A_max_size = std::min(A_max_size, gpu_max_buffer_size) / num_devices;

	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    d_ptrAInner[i] = clCreateBuffer(context, 
					    CL_MEM_READ_WRITE,
					    A_max_size*sizeof(REAL), 
					    NULL, &ciErrNum);
	    oclCheckErr(ciErrNum, "clCreateBuffer ptrAInner");
	  }
      }

      void compileGenAInnerKernel(int id, std::string kernel_src, cl_kernel* kernel) {
	cl_int err = CL_SUCCESS;
	
	std::string source2 = LaplaceInnerHeader();
	std::string l2dotfunction = InnerLTwoDotFunction();
	std::stringstream stream_program_src;

#ifdef USEOCL_NVIDIA
	stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
#endif
	stream_program_src << "#define REAL double" << std::endl;
	stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
	stream_program_src << l2dotfunction << std::endl;
	stream_program_src << source2 << std::endl;

	cl_uint sizec;
	clGetDeviceInfo(device_ids[0], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &sizec, 0);
	// std::cout << " CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV " << sizec << std::endl;
	size_t UseRegister = (sizec >= 3);//|| (dims <= 5);
	size_t dcon = UseRegister ? 0 : dims;

	stream_program_src << "__local REAL l2dotTemp["<<dcon*LSIZE<<"];" << std::endl;
	stream_program_src << "__local REAL gradTemp["<<dcon*LSIZE<<"];" << std::endl;

	stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
	stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
	stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;

	const char* typeREAL = "";

	stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;
	for(size_t d_inner = 0; d_inner < dims; d_inner++) {
	  if (d_inner == 0)
	    {
	      typeREAL = "REAL ";
	    } else {
	    typeREAL = "";
	  }
	  stream_program_src << typeREAL << "i_level =         ptrLevel["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
	  stream_program_src << typeREAL << "i_index =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;            
	  stream_program_src << typeREAL << "i_level_int =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;

	  stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
	  stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 1<<"]; " << std::endl;
	  stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;
	  const char* extra = "";
	  const char* extra2 = "";
	  if (d_inner < dcon) {
	    stream_program_src << "l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (l2dot(i_level"<<extra2<<",i_index"<<extra<<",i_level_int"<<extra<<", j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

	    stream_program_src << "gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (gradient(i_level"<<extra2<<",i_index"<<extra<<", j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
	  } else {
	    stream_program_src << "REAL l2dotreg_"<<d_inner<<" = (l2dot(i_level"<<extra2<<",i_index"<<extra<<",i_level_int"<<extra<<", j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

	    stream_program_src << "REAL gradreg_"<<d_inner<<" = (gradient(i_level"<<extra2<<",i_index"<<extra<<", j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
	  }

	}

	stream_program_src << "REAL res = 0.0;" << std::endl;
	stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
	typeREAL = "REAL ";

	for(unsigned d_outer = 0; d_outer < dims; d_outer++)
	  {
	    if (d_outer == 0)
	      {
		typeREAL = "REAL ";
	      } else {
	      typeREAL = "";
	    }

	    stream_program_src << typeREAL<< "element = 1.0;" << std::endl;
	    for(unsigned d_inner = 0; d_inner < dims; d_inner++)
	      {
		stream_program_src << "element *= ";
		if (d_outer == d_inner) {
		  if (d_inner < dcon) {
		    stream_program_src << "(gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
		  } else {
		    stream_program_src << "(gradreg_"<<d_inner<<");";
		  }
		} else {
		  if (d_inner < dcon) {
		    stream_program_src << "(l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
		  } else {
		    stream_program_src << "(l2dotreg_"<<d_inner<<");";
		  }
		}
		stream_program_src << std::endl;
	      }
	    stream_program_src << "res += ptrLambda["<<d_outer<<"] * element;" << std::endl;
	  }
	// 	stream_program_src << "ptrParResult[((get_group_id(0)+ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = res; " << std::endl;
	// 	stream_program_src << "ptrParResult[get_global_id(0)] = 1.1; " << std::endl;
// 	stream_program_src << "ptrParResult[(get_global_id(1) * LSIZE + get_local_id(0)) * get_global_size(0)  + get_group_id(0)*LSIZE + ConstantMemoryOffset*LSIZE + k] = (get_global_id(1) * LSIZE + get_local_id(0)) * get_global_size(0) + get_group_id(0)*LSIZE + ConstantMemoryOffset*LSIZE + k; " << std::endl;
	stream_program_src << "ptrParResult[(get_global_id(1) * LSIZE + get_local_id(0)) * overallMultOffset + get_group_id(0)*LSIZE + ConstantMemoryOffset*LSIZE + k] = res; " << std::endl;
// 	stream_program_src << "ptrParResult[(get_global_id(1) * LSIZE + get_local_id(0)) * "<<storageSizePadded<<" + get_group_id(0)*LSIZE + ConstantMemoryOffset*LSIZE + k] = res; " << std::endl;
	stream_program_src << "}" << std::endl; // k-loop
	stream_program_src << "}" << std::endl; // end of kernel

	std::string program_src = stream_program_src.str(); 
	const char* source3 = program_src.c_str();
	// #if PRINTOCL
	std::cout <<  source3 << std::endl;
	// #endif
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source3, NULL, &err);
	oclCheckErr(err, "clCreateProgramWithSource");
	char buildOptions[256];
	int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only -cl-strict-aliasing -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, dims);
	if (ierr < 0) {
	  printf("Error in Build Options");
	  exit(-1);
	}

	err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
	if (err != CL_SUCCESS)
	  {
	    std::cout << "OCL Error: compileMultKernel2DIndexesJInReg2 OpenCL Build Error. Error Code: " << err << std::endl;

	    size_t len;
	    char buffer[10000];

	    // get the build log
	    clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

	    std::cout << "--- Build Log ---" << std::endl << buffer << std::endl;
	  }
	oclCheckErr(err, "clBuildProgram");


	kernel[id] = clCreateKernel(program, kernel_src.c_str(), &err);
	oclCheckErr(err, "clCreateKernel");

	err |= clReleaseProgram(program);
	oclCheckErr(err, "clReleaseProgram");
  
      }
      void CompileGenAInnerKernels() {
	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    compileGenAInnerKernel(i, "multKernel", GenAInnerKernel);
	  }
      }

      void ExecGenAInner(REAL * ptrAlpha,
			 REAL * ptrA,
			 REAL * lcl_q,
			 REAL * lcl_q_inv,
			 REAL *ptrLevel,
			 REAL *ptrIndex,
			 REAL *ptrLevel_int,
			 size_t argStorageSize,
			 size_t argStorageDim) {

	cl_int ciErrNum = CL_SUCCESS;
	cl_event GPUExecution[NUMDEVS];
  
	size_t idx = 0;
	for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
	  ptrLcl_qInner[idx++] = lcl_q[d_outer];
	  ptrLcl_qInner[idx++] = lcl_q_inv[d_outer];
	}


	for(size_t i = 0; i < num_devices; i++) {
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qInner[i], CL_FALSE, 0,
					   lcl_q_size*sizeof(REAL), ptrLcl_qInner, 0, 0, &GPUExecution[i]);
	}
	//	clWaitForEvents(num_devices, GPUDoneLcl);

	{
	  size_t multglobalworksize[2];
	  size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, A_max_size / (storageSizePadded));
// 	  std::cout << " A_max_size " << A_max_size << std::endl;
 
// 	  std::cout << " storageSizePaddedStep " << storageSizePaddedStep << std::endl;

	  multglobalworksize[0] = std::min(storageSizePadded,storageSizePaddedStep);
	  multglobalworksize[1] = storageSizePadded/LSIZE;
	  size_t multglobal = storageSizePadded / num_devices;


	  for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
	    multglobalworksize[0] =  std::min(multglobalworksize[0], multglobal-overallMultOffset);

// 	    std::cout << " multglobalworksize[0] " << multglobalworksize[0] << std::endl;
	    for(unsigned int i = 0; i < num_devices; i++) {
	      size_t overallMultOffset2 = multglobalworksize[0];
	      ciErrNum |= clSetKernelArg(GenAInnerKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
	    }
	    size_t constantglobalworksize[2];
	    size_t constantlocalworksize[2];
	    size_t constantglobal = multglobalworksize[0];
	    constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,constantglobal);
	    constantglobalworksize[1] = multglobalworksize[1];
	    constantlocalworksize[0] = LSIZE;
	    constantlocalworksize[1] = 1;
	    for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	      constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);

// 	      std::cout << "constantglobalworksize[0] " << constantglobalworksize[0]  << std::endl;
	      // 	      std::cout << "constantglobalworksize[1]" << constantglobalworksize[1] << std::endl;
	
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
						 d_ptrLevelIndexLevelintconInner[i], 
						 CL_TRUE, 0 ,
						 constantglobalworksize[0]*3*dims*sizeof(REAL), 
						 ptrLevelIndexLevelintInner+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
						 1,
						 &GPUExecution[i] , NULL);
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferGenAL253");
		size_t jj = (ConstantMemoryOffset) / LSIZE;

		ciErrNum |= clSetKernelArg(GenAInnerKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL302");

		ciErrNum = clEnqueueNDRangeKernel(command_queue[i], 
						  GenAInnerKernel[i], 
						  2, 0, 
						  constantglobalworksize, 
						  constantlocalworksize,
						  0, NULL, &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	      }
	    }
	    
	    for(unsigned int i = 0; i < num_devices; i++) {
	      ciErrNum |= clFinish(command_queue[i]);
	      oclCheckErr(ciErrNum, "clFinishLapIL355");
	    }
// 	    std::cout << " overallMultOffset " << overallMultOffset << std::endl;
	    size_t buffer_origin[3] = {0,0,0};
	    size_t host_origin[3] = {overallMultOffset*sizeof(REAL),0,0};
	    size_t ReadSize = std::min(multglobalworksize[0], storageSize-overallMultOffset);
	    size_t region[3] = {ReadSize*sizeof(REAL), storageSize,1};
	    size_t buffer_row_pitch = multglobalworksize[0]*sizeof(REAL);
	    size_t buffer_slice_pitch = storageSizePadded * buffer_row_pitch;

	    size_t host_row_pitch = storageSize*sizeof(REAL);
	    size_t host_slice_pitch = storageSize * host_row_pitch;

	    for(unsigned int i = 0;i < num_devices; i++) 
	      {    
// 		clEnqueueReadBuffer(command_queue[i], d_ptrAInner[i], CL_TRUE, 0,
// 				    storageSizePadded*storageSizePadded*sizeof(REAL), ptrAInner, 0, NULL, NULL);
		ciErrNum |= clEnqueueReadBufferRect(command_queue[i], 
						    d_ptrAInner[i], CL_FALSE,
						    buffer_origin,
						    host_origin,
						    region,
						    buffer_row_pitch,
						    buffer_slice_pitch,
						    host_row_pitch,
						    host_slice_pitch,
						    ptrA,
						    0, NULL, &GPUExecution[i]);
						
		oclCheckErr(ciErrNum, "clFinishLapIL355");
	      }
	    clWaitForEvents(num_devices, GPUExecution);

	  }
	}

#if TOTALTIMING
	CounterLaplaceInner += 1.0;
#endif	
	for(size_t i = 0; i < num_devices; i++) {
	  clReleaseEvent(GPUExecution[i]);
	}  
      }

    }

    using namespace oclpdekernels;
    void OCLPDEKernels::RunOCLKernelGenAInner(sg::base::DataVector& alpha,
					      REAL * ptrA,
					      REAL * lcl_q,
					      REAL * lcl_q_inv,
					      REAL * ptrLevel,
					      REAL * ptrIndex,
					      REAL * ptrLevel_int,
					      REAL * ptrLambda,
					      size_t argStorageSize,
					      size_t argStorageDim,
					      sg::base::GridStorage * storage) {
      //std::cout << "EXEC RunOCLKernelInner" << std::endl;
      myStopwatch->start();
      if (isVeryFirstTime) {
	StartUpGPU();
      } 
      if (isFirstTimeLaplaceInner && isFirstTimeLTwoDotInner) {
	SetBuffersInner(ptrLevel,
			ptrIndex,
			ptrLevel_int,
			argStorageSize,
			argStorageDim,storage);

      }
      if (isFirstTimeLaplaceInner) {
	SetLambdaBufferLaplaceInner(ptrLambda,
				    argStorageDim);
	SetABufferGenAInner();
	CompileGenAInnerKernels(); 
	SetArgumentsGenAInner();
	isVeryFirstTime = 0;
	isFirstTimeLaplaceInner = 0;
      }
      LaplaceInnerStartupTime += myStopwatch->stop();

      myStopwatch->start();
      ExecGenAInner(alpha.getPointer(), ptrA,lcl_q,lcl_q_inv, ptrLevel, ptrIndex, ptrLevel_int, argStorageSize, argStorageDim);
      LaplaceInnerExecTime += myStopwatch->stop();
      //std::cout << "EXIT RunOCLKernelInner" << std::endl;
    }
  }
}

