#ifndef USE_MPI
#include "OCLLTwoDotLaplaceInner.hpp"

namespace sg {
  namespace parallel {
    namespace oclpdekernels {

      cl_kernel SymmetricLTwoDotLaplaceInnerKernel[NUMDEVS];
      cl_kernel SymmetricReduceInnerKernel[NUMDEVS];
      cl_mem d_ptrSymmetricResultInner[NUMDEVS];
      REAL * ptrSymmetricResultInner;
      size_t symResult_size;
      size_t symResultY_size;
      size_t symResultX_size;
      size_t storageSizePaddedStep;
      size_t * GlobalSize_ii;
      size_t * GlobalOffset_ii;


      void SetSymmetricBuffersInner() {
	cl_int ciErrNum = CL_SUCCESS;
	storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * LSIZE);
	std::cout << " storageSizePaddedStep " << storageSizePaddedStep << std::endl;
	size_t minStepSize = std::min(storageSizePadded,storageSizePaddedStep);
	symResultY_size = std::min(constant_buffer_iterations_noboundary,minStepSize);
	size_t acc = 0;
	size_t count = 0;
	while (acc < storageSizePadded) {
	  count++;
	  acc += symResultY_size;
	}
	std::cout << " count " << count << " symResultY_size " << symResultY_size << std::endl;
	GlobalSize_ii = (size_t*)calloc(count, sizeof(size_t));
	GlobalOffset_ii = (size_t*)calloc(count, sizeof(size_t));
	
	acc = storageSizePadded;
	size_t count2 = 0;
	size_t offset = 0;
	while (0 < acc) {
	  std::cout << " acc " << acc << std::endl;
	  GlobalSize_ii[count2] = acc;
	  GlobalOffset_ii[count2++] = offset;
	  acc -= std::min(symResultY_size, acc);
	  offset += std::min(symResultY_size, storageSizePadded-offset);
	}
	for (size_t i = 0; i < count; i++) {
	  GlobalSize_ii[i] /= LSIZE;
	  GlobalOffset_ii[i] /= LSIZE;
	  std::cout << " GlobalSize_ii["<<i<<"] = "<< GlobalSize_ii[i] << std::endl;
	  std::cout << " GlobalOffset_ii["<<i<<"] = "<< GlobalOffset_ii[i] << std::endl;
	}

	symResultX_size = storageSizePadded/LSIZE;

	symResult_size = symResultY_size * symResultX_size;
	ptrSymmetricResultInner = (REAL*)calloc(symResult_size, sizeof(REAL));
	for(size_t i=0; i < num_devices; ++i) 
	  {
	    d_ptrSymmetricResultInner[i] = clCreateBuffer(context, CL_MEM_READ_WRITE ,
							  symResult_size*sizeof(REAL), NULL,&ciErrNum);
	    oclCheckErr(ciErrNum, "clCreateBuffer ptrSymmetricResult");
	  }

      }
      std::string SymmetricReduceInnerKernelStr() {
	std::stringstream stream_program_src;
	stream_program_src <<   "__kernel void ReduceInnerKernel(__global  REAL* ptrResult,    "  << std::endl;
	stream_program_src <<	"                                __global  REAL* ptrSymmetricResult, "  << std::endl;
	stream_program_src <<	"                		 ulong jj_offset,      "  << std::endl;
	stream_program_src <<	"                		 ulong num_groups,      "  << std::endl;
	stream_program_src <<	"                		 ulong ii_offset)      "  << std::endl;
	stream_program_src <<	"{							       "  << std::endl;
	stream_program_src <<	"unsigned j = get_global_id(0);				       "  << std::endl;
	stream_program_src <<	"REAL res = 0.0;					       "  << std::endl;
	stream_program_src <<	"for (unsigned k = ii_offset; k < num_groups; k++) {		       "  << std::endl;
	stream_program_src <<	"	res += ptrSymmetricResult[k*get_global_size(0) + j];         "  << std::endl;
	stream_program_src <<	"}							       "  << std::endl;
	stream_program_src <<	"ptrResult[jj_offset+j] += res;		       "  << std::endl;
	stream_program_src <<	"}							       "  << std::endl;
	return stream_program_src.str();
      }	

      void CompileSymmetricReduceInner(int id, std::string kernel_src, cl_kernel* kernel) {

	cl_int err = CL_SUCCESS;
	std::string source2 = SymmetricReduceInnerKernelStr();
  
	std::stringstream stream_program_src;

#ifdef USEOCL_NVIDIA
	stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
#endif
	stream_program_src << "#define REAL double" << std::endl;
	stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
	stream_program_src << source2 << std::endl;
	std::string program_src = stream_program_src.str(); 
	const char* source3 = program_src.c_str();
	//  std::cout << "SOURCE " << source2 << std::endl;
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source3, NULL, &err);
	oclCheckErr(err, "clCreateProgramWithSource");
	char buildOptions[256];
	int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DINNERSTORAGEPAD=%lu -DDIMS=%lu -cl-finite-math-only -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, 0L, dims);
	if (ierr < 0) {
	  printf("Error in Build Options");
	  exit(-1);
	}
	err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
	if (err != CL_SUCCESS)
	  {
	    std::cout << "OCL Error: compileReduceInner OpenCL Build Error. Error Code: " << err << std::endl;

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


      std::string SymmetricLTwoDotLaplaceInnerHeader() {
	std::stringstream stream_program_src;
	stream_program_src << "__kernel void multKernel(__global  REAL* ptrLevel,			"  << std::endl;
	stream_program_src << "			 	__constant  REAL* ptrLevelIndexLevelintcon,	"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrIndex, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrLevel_int, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrAlpha, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrParResult, 			"  << std::endl;
	stream_program_src << "			 	__constant  REAL* ptrLcl_q,			"  << std::endl;
	stream_program_src << "			 	__constant  REAL* ptrLambda,			"  << std::endl;
	stream_program_src << "			 	ulong overallMultOffset,   			"  << std::endl;
	stream_program_src << "                        	ulong ConstantMemoryOffset,   			"  << std::endl;
	stream_program_src << "                        	REAL TimestepCoeff,		                "  << std::endl;
	stream_program_src << "                        	__global  REAL* ptrSymmetricResult)                "  << std::endl;
	stream_program_src << "{									"  << std::endl;
	return stream_program_src.str(); 
      }
      void SetArgumentsSymmetricLTwoDotLaplaceInner() {

	cl_int ciErrNum = CL_SUCCESS;
	int counter = 0;

	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelIndexLevelintconInner[i]);

	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intInner[i]);

	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLambdaInner[i]);

	    ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], 11, sizeof(cl_mem), (void *) &d_ptrSymmetricResultInner[i]);

	    oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");
	    counter = 0;
	    ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultInner[i]);
	    ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrParResultInner[i]);
	    counter = 0;
	    oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");
	    ciErrNum |= clSetKernelArg(SymmetricReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultInner[i]);
	    ciErrNum |= clSetKernelArg(SymmetricReduceInnerKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrSymmetricResultInner[i]);
	    counter = 0;
	    oclCheckErr(ciErrNum, "clSetKernelArg1 Kernel Construct");
	  }

      }
      void CompileSymmetricLTwoDotLaplaceInner(int id, std::string kernel_src, cl_kernel* kernel) {
	cl_int err = CL_SUCCESS;
	
	std::string source2 = SymmetricLTwoDotLaplaceInnerHeader();
	std::string l2dotfunction = InnerLTwoDotFunction();
	std::string gradientfunction = InnerGradientFunction();
	std::stringstream stream_program_src;
	

#ifdef USEOCL_NVIDIA
	stream_program_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
#endif
	stream_program_src << "#define REAL double" << std::endl;
	stream_program_src << "#define LSIZE "<<LSIZE << std::endl;
	stream_program_src << l2dotfunction << std::endl;
	stream_program_src << gradientfunction << std::endl;
	stream_program_src << source2 << std::endl;
	stream_program_src << "__local REAL alphaTemp["<<LSIZE<<"];" << std::endl;
	stream_program_src << "__local REAL SymmetricResult["<<LSIZE<<"];" << std::endl;
	cl_uint sizec;
	clGetDeviceInfo(device_ids[0], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &sizec, 0);

	size_t keplerUseRegister = (sizec >= 3);

	size_t fermiUseRegister = (dims <= 5);

#ifdef USEOCL_INTEL
	keplerUseRegister = 1;
	fermiUseRegister = 0;
#endif
	size_t dcon = (fermiUseRegister || keplerUseRegister) ? 0 : dims;

	stream_program_src << "__local REAL l2dotTemp["<<dcon*LSIZE<<"];" << std::endl;
	stream_program_src << "__local REAL gradTemp["<<dcon*LSIZE<<"];" << std::endl;

	stream_program_src << "ptrLevel += get_local_id(0);" << std::endl;
	stream_program_src << "ptrIndex += get_local_id(0);" << std::endl;
	stream_program_src << "ptrLevel_int += get_local_id(0);" << std::endl;

	stream_program_src << "alphaTemp[get_local_id(0)]   = ptrAlpha[get_global_id(0) + overallMultOffset + ConstantMemoryOffset*LSIZE];" << std::endl;

	stream_program_src << "REAL SymmetricRes_ii = 0.0;" << std::endl;
	stream_program_src << "unsigned jj = 0;" << std::endl;
	stream_program_src << "unsigned ii = 0;" << std::endl;
	stream_program_src << "REAL alphaReg_ii = ptrAlpha[(get_global_id(1)*LSIZE+get_local_id(0))];" << std::endl;
	stream_program_src << "REAL SymmetricRes_jj = 0.0;" << std::endl;
	const char* typeREAL = "";


	if (keplerUseRegister) {
	  const char* typeREAL2 = "REAL ";
	  for(size_t d_inner = 0; d_inner < dims; d_inner++) {
	    stream_program_src << typeREAL2 << "i_level_"<<d_inner<<" =         ptrLevel["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
	    stream_program_src << typeREAL2 << "i_index_"<<d_inner<<" =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;            
	    stream_program_src << typeREAL2 << "i_level_int_"<<d_inner<<" =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
	  }
	}

	stream_program_src << "for (unsigned k = 0; k < "<<LSIZE<<"; k++) {" << std::endl;
	stream_program_src << "REAL SymmetricRes = 0.0;" << std::endl;
	for(size_t d_inner = 0; d_inner < dims; d_inner++) {
	  std::stringstream registerId;
	  if (d_inner == 0)
	    {
	      typeREAL = "REAL ";
	    } else {
	    typeREAL = "";
	  }
	  registerId << "_"<<d_inner;
	  std::string registerIdStr = registerId.str();
	  if (!keplerUseRegister) {
	    const char* typeREAL2 = "REAL ";
	    stream_program_src << typeREAL2 << "i_level"<<registerIdStr<<" =         ptrLevel["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
	    stream_program_src << typeREAL2 << "i_index"<<registerIdStr<<" =         ptrIndex["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;            
	    stream_program_src << typeREAL2 << "i_level_int"<<registerIdStr<<" =     ptrLevel_int["<<d_inner*storageSizePadded<<" + (get_global_id(1))*LSIZE];" << std::endl;
	  }

	  stream_program_src << typeREAL << "j_levelreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3<<"]; " << std::endl;
	  stream_program_src << typeREAL << "j_indexreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 1<<"]; " << std::endl;
	  stream_program_src << typeREAL << "j_level_intreg = ptrLevelIndexLevelintcon[(get_group_id(0)*LSIZE+k)*"<<dims*3<<" + "<<d_inner*3 + 2<<"]; " << std::endl;

	  if (d_inner < dcon) {
	    stream_program_src << "l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (l2dot(i_level"<<registerIdStr<<",\
             i_index"<<registerIdStr<<",\
             i_level_int"<<registerIdStr<<", \
            j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

	    stream_program_src << "gradTemp["<<d_inner*LSIZE<<" + get_local_id(0)] = (gradient(i_level"<<registerIdStr<<",\
               i_index"<<registerIdStr<<", \
               j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
	  } else {
	    stream_program_src << "REAL l2dotreg_"<<d_inner<<" = \
	                 (l2dot(i_level"<<registerIdStr<<",\
	                        i_index"<<registerIdStr<<",\
	                        i_level_int"<<registerIdStr<<",\
	                        j_levelreg,j_indexreg,j_level_intreg,"<< "ptrLcl_q["<<(d_inner+1)*2-2<<"]));" << std::endl;

	    stream_program_src << "REAL gradreg_"<<d_inner<<" = \
	               (gradient(i_level"<<registerIdStr<<",\
	                         i_index"<<registerIdStr<<", \
	                         j_levelreg,j_indexreg,"<< "ptrLcl_q["<<(d_inner+1)*2-1<<"]));" << std::endl;
	  }

	}
	stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
	typeREAL = "REAL ";
	stream_program_src << typeREAL<< "alphaReg_jj = alphaTemp["<<"k"<<"];" << std::endl;

	for(unsigned d_outer = 0; d_outer < dims; d_outer++)
	  {
	    if (d_outer == 0)
	      {
		typeREAL = "REAL ";
	      } else {
	      typeREAL = "";
	    }

	    stream_program_src << typeREAL << "element = 1.0;" << std::endl;
	    for(unsigned d_inner = 0; d_inner < dims; d_inner++)
	      {
		if (d_outer == d_inner) {

		} else {
		  stream_program_src << "element *= ";
		  if (d_inner < dcon) {
		    stream_program_src << "(l2dotTemp["<<d_inner*LSIZE<<" + get_local_id(0)]);";
		  } else {
		    stream_program_src << "(l2dotreg_"<<d_inner<<");";
		  }
		}
		stream_program_src << std::endl;
	      }
	  
	    if (d_outer < dcon) {
	      
	      stream_program_src << "SymmetricRes += TimestepCoeff*ptrLambda["<<d_outer<<"] * element* gradTemp["<<d_outer*LSIZE<<" + get_local_id(0)];" << std::endl;
	    } else {
	      
	      stream_program_src << "SymmetricRes += TimestepCoeff*ptrLambda["<<d_outer<<"] * element* gradreg_"<<d_outer<<";" << std::endl;
	    }
	      
	  }
	if ((dims-1) < dcon) {
	  stream_program_src << "SymmetricRes += element * l2dotTemp["<<(dims-1)*LSIZE<<" + get_local_id(0)]; " << std::endl;
	} else {
	  stream_program_src << "SymmetricRes += element * (l2dotreg_"<<(dims-1)<<"); " << std::endl;
 	}
	// 	stream_program_src << "SymmetricRes_ii += SymmetricRes * alphaReg_jj; " << std::endl;
	stream_program_src << "jj = get_global_id(0) + overallMultOffset + ConstantMemoryOffset*LSIZE; " << std::endl;
	stream_program_src << "ii = (get_global_id(1)*LSIZE+get_local_id(0)); " << std::endl;
	
	stream_program_src << "SymmetricRes_ii += SymmetricRes * alphaReg_jj * (jj<=ii); " << std::endl;
	stream_program_src << "SymmetricResult[get_local_id(0)] = SymmetricRes * alphaReg_ii * (jj < ii); " << std::endl;
	stream_program_src << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	stream_program_src << "if (get_local_id(0) == k) {                 " << std::endl;
	stream_program_src << "      for (unsigned iii = 0; iii < LSIZE; iii++) {  " << std::endl;
	stream_program_src << "            SymmetricRes_jj += SymmetricResult[iii]; " << std::endl;
	stream_program_src << "     } " << std::endl;
	stream_program_src << "} " << std::endl;

	stream_program_src << "}" << std::endl;
	// 	stream_program_src << "ptrSymmetricResult[(get_global_id(0))*get_global_size(1) + get_global_id(1) ] = SymmetricRes_jj; " << std::endl;

	// Trans
	stream_program_src << "ptrSymmetricResult[get_global_id(1)*get_global_size(0) + get_global_id(0)] = SymmetricRes_jj; " << std::endl;

	stream_program_src << "ptrParResult[((get_group_id(0)+0*ConstantMemoryOffset)*get_global_size(1) + get_global_id(1))*"<<LSIZE<<" + get_local_id(0)] = SymmetricRes_ii; " << std::endl;
	// 	stream_program_src << "ptrParResult[(get_local_id(0) + get_global_id(1)*LSIZE) * get_global_size(0)/LSIZE + get_group_id(0)] = SymmetricRes_ii; " << std::endl;
	stream_program_src << "}" << std::endl;

	std::string program_src = stream_program_src.str(); 
	const char* source3 = program_src.c_str();
	std::cout <<  source3 << std::endl;
#if PRINTOCL
	std::cout <<  source3 << std::endl;
#endif
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source3, NULL, &err);
	oclCheckErr(err, "clCreateProgramWithSource");
	char buildOptions[256];
	int ierr = snprintf(buildOptions, sizeof(buildOptions), "-DSTORAGE=%lu -DSTORAGEPAD=%lu -DNUMGROUPS=%lu -DDIMS=%lu -cl-finite-math-only -cl-fast-relaxed-math ", storageSize, storageSizePadded, num_groups, dims);
	if (ierr < 0) {
	  printf("Error in Build Options");
	  exit(-1);
	}

	err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
	if (err != CL_SUCCESS)
	  {
	    std::cout << "OCL Error: compileLTwoDotLaplaceinner OpenCL Build Error. Error Code: " << err << std::endl;

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
      void CompileSymmetricLTwoDotLaplaceInnerKernels() {
	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    CompileSymmetricLTwoDotLaplaceInner(i, "multKernel", SymmetricLTwoDotLaplaceInnerKernel);
	    CompileReduceInner(i, "ReduceInnerKernel", ReduceInnerKernel);
	    CompileSymmetricReduceInner(i, "ReduceInnerKernel", SymmetricReduceInnerKernel);
	  }
      }

      void ExecSymmetricLTwoDotLaplaceInner(REAL * ptrAlpha,
					    REAL * ptrResult,
					    REAL * lcl_q,
					    REAL * lcl_q_inv,
					    REAL * ptrLevel,
					    REAL * ptrIndex,
					    REAL * ptrLevel_int) {

	cl_int ciErrNum = CL_SUCCESS;
	cl_event GPUDone[NUMDEVS];
	cl_event GPUDoneLcl[NUMDEVS];
	cl_event GPUExecution[NUMDEVS];
  
	size_t idx = 0;
	for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
	  ptrLcl_qInner[idx++] = lcl_q[d_outer];
	  ptrLcl_qInner[idx++] = lcl_q_inv[d_outer];
	}


	for(size_t i = 0; i < num_devices; i++) {

	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, 0,
					   storageSize*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qInner[i], CL_FALSE, 0,
					   lcl_q_size*sizeof(REAL), ptrLcl_qInner, 0, 0, &GPUDoneLcl[i]);
	}
	clWaitForEvents(num_devices, GPUDone);
	clWaitForEvents(num_devices, GPUDoneLcl);
	for(size_t i = 0; i < num_devices; i++) {
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrResultInner[i], 
					   CL_FALSE, 0,
					   storageSizePadded*sizeof(REAL), 
					   ptrResultZero, 0, 0, &GPUDone[i]);
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaInner[i], CL_FALSE, storageSize*sizeof(REAL),
					   alphaend_size*sizeof(REAL), ptrAlphaEndInner, 0, 0, &GPUExecution[i]);
	}
	oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");
	clWaitForEvents(num_devices, GPUDone);

	{
	  size_t multglobalworksize[2];
	  // 	  size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * LSIZE);
	  multglobalworksize[0] = std::min(storageSizePadded,storageSizePaddedStep);
	  multglobalworksize[1] = storageSizePadded/LSIZE;
	  size_t multglobal = storageSizePadded / num_devices;
	  size_t global_i = multglobalworksize[1];
	  size_t kernelcounter = 0;

	  for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
	    multglobalworksize[0] =  std::min(multglobalworksize[0], multglobal-overallMultOffset);
	    for(unsigned int i = 0; i < num_devices; i++) {
	      size_t overallMultOffset2 = overallMultOffset + i*multglobal;
	      ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
	    }
	    oclCheckErr(ciErrNum, "clSetKernelArgL246");
	    size_t constantglobalworksize[2];
	    size_t constantlocalworksize[2];
	    size_t constantgridoffset[2];
	    size_t constantglobal = multglobalworksize[0];
	    constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,constantglobal);

	    constantglobalworksize[1] = multglobalworksize[1];
	    constantlocalworksize[0] = LSIZE;
	    constantlocalworksize[1] = 1;
	    constantgridoffset[0] = 0;
	    
	    for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	      constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);
	      // 	      std::cout << " constantglobalworksize[0], ConstantMemoryOffset " << constantglobalworksize[0] << ", " << ConstantMemoryOffset << std::endl;
	      constantgridoffset[1] = GlobalOffset_ii[kernelcounter];
 	      constantglobalworksize[1] = GlobalSize_ii[kernelcounter++];

	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
						 d_ptrLevelIndexLevelintconInner[i], 
						 CL_TRUE, 0 ,
						 constantglobalworksize[0]*3*dims*sizeof(REAL), 
						 ptrLevelIndexLevelintInner+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
						 1,
						 &GPUExecution[i] , NULL);
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL306");
		size_t jj = (ConstantMemoryOffset) / LSIZE;
		
		ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
		ciErrNum |= clSetKernelArg(SymmetricLTwoDotLaplaceInnerKernel[i], 10, sizeof(REAL), (void *) &TimestepCoeff);

		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL302");
		ciErrNum = clEnqueueNDRangeKernel(command_queue[i], 
						  SymmetricLTwoDotLaplaceInnerKernel[i], 
						  2, constantgridoffset, 
						  constantglobalworksize, 
						  constantlocalworksize,
						  0, NULL, &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	      }
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clFinish(command_queue[i]);
		oclCheckErr(ciErrNum, "clFinishLapIL105");
	      }

	      // 	      std::cout << " constantgridoffset[0] " << constantgridoffset[0] << std::endl;
	      // 	      std::cout << " constantglobalworksize[0] " << constantglobalworksize[0] << std::endl;
	      // 	      std::cout << " constantgridoffset[1] " << constantgridoffset[1] << std::endl;
	      // 	      std::cout << " constantglobalworksize[1] " << constantglobalworksize[1] << std::endl << std::endl;


	      size_t overallReduceOffset2 = overallMultOffset + ConstantMemoryOffset;
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clSetKernelArg(SymmetricReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset2);
		oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");
		size_t newnum_groups = constantgridoffset[1] + constantglobalworksize[1] ;
		ciErrNum |= clSetKernelArg(SymmetricReduceInnerKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
		oclCheckErr(ciErrNum, "clSetKernelArgLapIL340");
		ciErrNum |= clSetKernelArg(SymmetricReduceInnerKernel[i], 4, sizeof(cl_ulong), (void *) &constantgridoffset[1]);
		oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");

		size_t reduceglobalworksize2[] = {constantglobalworksize[0], 1};
		size_t reduceglobaloffset[] = {0,0};
		size_t local2[] = {LSIZE,1};
		ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
						   SymmetricReduceInnerKernel[i], 
						   2, reduceglobaloffset, 
						   reduceglobalworksize2, 
						   local2,
						   0, NULL, &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
	      }
// 	      // Need extra reduce here. 
// 	      for(unsigned int i = 0;i < num_devices; i++) {
// 		ciErrNum |= clEnqueueReadBuffer(command_queue[i], 
// 						d_ptrSymmetricResultInner[i], CL_FALSE, 0,
// 						symResult_size*sizeof(REAL), 
// 						ptrSymmetricResultInner, 
// 						0, NULL, &GPUDone[i]);

// 	      }
// 	      oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
// 	      ciErrNum |= clWaitForEvents(num_devices, GPUDone);

	      // 	      std::cout << "SYMMETRIC: " << std::endl;
	      // 	      std::cout << " start_i " << reduOffset << std::endl;
	      // 	      std::cout << " end_i " << std::min(reduOffset+constantglobalworksize[0], storageSize) << std::endl;
	      // 	      std::cout << " start_j " << constantgridoffset[1] << std::endl;
	      // 	      std::cout << " end_j " << constantgridoffset[1]+constantglobalworksize[1] << std::endl << std::endl;
		

// 	      size_t reduOffset = overallMultOffset + ConstantMemoryOffset;
// 	      for (size_t i = constantgridoffset[1]; i < constantgridoffset[1]+constantglobalworksize[1]; i++) {
// 		for (size_t j = reduOffset; j < std::min(reduOffset+constantglobalworksize[0], storageSize); j++) {
// 		  ptrResult[j] += ptrSymmetricResultInner[i*constantglobalworksize[0] + (j-reduOffset)];
// 		}
// 	      }

// 	      ciErrNum |= clEnqueueReadBuffer(command_queue[0], d_ptrParResultInner[0], CL_TRUE, 0,
// 					      par_result_max_size*sizeof(REAL), ptrParResultInner, 0, NULL, NULL);
// 	      oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL161");

	      // 	      std::cout << "NORMAL: " << std::endl;
	      // 	      std::cout << " start_i " << constantgridoffset[1]*LSIZE << std::endl;
	      // 	      std::cout << " end_i " << storageSize << std::endl;
	      // 	      std::cout << " start_j " << 0 << std::endl;
	      // 	      std::cout << " end_j " << constantglobalworksize[0]/LSIZE << std::endl << std::endl;

	      size_t overallReduceOffset = 0;//overallMultOffset*LSIZE;
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
		oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");
		size_t newnum_groups = constantglobalworksize[0] / LSIZE ;
		ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
		oclCheckErr(ciErrNum, "clSetKernelArgLapIL340");

		size_t reduceglobalworksize2[] = {constantglobalworksize[1]*LSIZE, 1};
		size_t reduceglobaloffset[] = {constantgridoffset[1]*LSIZE,0};
		size_t local2[] = {LSIZE,1};
		ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
						   ReduceInnerKernel[i], 
						   2, reduceglobaloffset, reduceglobalworksize2, 
						   local2,
						   0, NULL, &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
	      }
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clFinish(command_queue[i]);
		oclCheckErr(ciErrNum, "clFinishLapIL355");
	      }
	      // 	      for (size_t j = 0; j < constantglobalworksize[0]/LSIZE; j++) {
	      // 		for (size_t i = constantgridoffset[1]*LSIZE; i < storageSize; i++) {
	      // 		  ptrResult[i] += ptrParResultInner[j* constantglobalworksize[1]*LSIZE  +i];
	      // 		}
	      // 	      }

	    }


	    
	    // 	    for(unsigned int i = 0; i < num_devices; i++) {
	    // 	      ciErrNum |= clFinish(command_queue[i]);
	    // 	      oclCheckErr(ciErrNum, "clFinishLapIL105");
	    // 	    }
	    // 	      LTwoDotLaplaceInnerProfilingAcc += AccumulateTiming(GPUExecution, 0);
	    // 	      LTwoDotLaplaceInnerProfilingWait += AccumulateWaiting(GPUExecution, 0);
	    
	    //REDUCE
	    // 	    size_t overallReduceOffset = 0;//overallMultOffset*LSIZE; 
	    // 	    for(unsigned int i = 0; i < num_devices; i++) {

	    // 	      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	    // 	      oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");
	    // 	      size_t newnum_groups = multglobalworksize[0] / LSIZE ;
	    // 	      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
	    // 	      oclCheckErr(ciErrNum, "clSetKernelArgLapIL340");

	    // 	      size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	    // 	      size_t local2[] = {LSIZE,1};
	    // 	      ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
	    // 						 ReduceInnerKernel[i], 
	    // 						 2, 0, reduceglobalworksize2, 
	    // 						 local2,
	    // 						 0, NULL, &GPUExecution[i]);
	    // 	      oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
	    // 	    }
	    // 	    for(unsigned int i = 0; i < num_devices; i++) {
	    // 	      ciErrNum |= clFinish(command_queue[i]);
	    // 	      oclCheckErr(ciErrNum, "clFinishLapIL355");
	    // 	    }

	  }
	}

	//std::cout << " num_devices " << num_devices << std::endl;
	if (num_devices > 1) {
	    
	  for(unsigned int i = 0;i < num_devices; i++) 
	    {    
	      ciErrNum |= clEnqueueReadBuffer(command_queue[i], 
					      d_ptrResultInner[i], CL_FALSE, 0,
					      storageSize*sizeof(REAL), 
					      ptrResultTemp + i * storageSizePadded, 
					      0, NULL, &GPUDone[i]);

	    }
	  oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
	  ciErrNum |= clWaitForEvents(num_devices, GPUDone);
	  oclCheckErr(ciErrNum, "clWaitForEventsLapIL2147");

	  for (size_t j = 0; j < num_devices; j++) {
	    for (size_t i = 0; i < storageSize; i++) {
	      ptrResult[i] += ptrResultTemp[j*storageSizePadded + i];

	    }
	  }

	} else {
	  for(unsigned int i = 0;i < num_devices; i++) 
	    {    
	      // 	      clEnqueueReadBuffer(command_queue[i], d_ptrResultInner[i], CL_FALSE, 0,
	      // 				  storageSize*sizeof(REAL), ptrResult, 0, NULL, &GPUDone[i]);

	      ciErrNum |= clEnqueueReadBuffer(command_queue[0], d_ptrResultInner[0], CL_TRUE, 0,
					      storageSize*sizeof(REAL), ptrResultPinnedInner, 0, NULL, NULL);
	      oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL161");
	    }
	  clWaitForEvents(num_devices, GPUDone);
	  for( size_t i = 0; i < storageSize; i++) {
	    ptrResult[i] += ptrResultPinnedInner[i];
	  }

	}
	

#if TOTALTIMING
	CounterLTwoDotLaplaceInner += 1.0;
#endif	
	for(size_t i = 0; i < num_devices; i++) {
	  clReleaseEvent(GPUExecution[i]);
	  clReleaseEvent(GPUDone[i]);
	  clReleaseEvent(GPUDoneLcl[i]);
	}  
      }

    }

    using namespace oclpdekernels;
    void OCLPDEKernels::RunOCLKernelSymmetricLTwoDotLaplaceInner(sg::base::DataVector& alpha,
								 sg::base::DataVector& result,
								 REAL * lcl_q,
								 REAL * lcl_q_inv,
								 REAL * ptrLevel,
								 REAL * ptrIndex,
								 REAL * ptrLevel_int,
								 REAL * ptrLambda,
								 size_t argStorageSize,
								 size_t argStorageDim,
								 sg::base::GridStorage * storage,
								 REAL tsCoeff) {

      TimestepCoeff = tsCoeff;

      if (isVeryFirstTime) {
	StartUpGPU();
      } 
      if (isFirstTimeLaplaceInner && 
	  isFirstTimeLTwoDotInner && 
	  isFirstTimeLTwoDotLaplaceInner) {
	SetBuffersInner(ptrLevel,
			ptrIndex,
			ptrLevel_int,
			argStorageSize,
			argStorageDim,storage);

      }
      if (isFirstTimeLaplaceInner && isFirstTimeLTwoDotLaplaceInner) {
	SetLambdaBufferLaplaceInner(ptrLambda,
				    argStorageDim);
	
      }

      if (isFirstTimeLTwoDotLaplaceInner) {
	CompileSymmetricLTwoDotLaplaceInnerKernels(); 
	SetSymmetricBuffersInner();
	SetArgumentsSymmetricLTwoDotLaplaceInner();
	isVeryFirstTime = 0;
	isFirstTimeLTwoDotLaplaceInner = 0;
      }


      myStopwatch->start();
      ExecSymmetricLTwoDotLaplaceInner(alpha.getPointer(), 
				       result.getPointer(), 
				       lcl_q, lcl_q_inv,
				       ptrLevel, ptrIndex, 
				       ptrLevel_int);
      double runtime = myStopwatch->stop();
      LTwoDotLaplaceInnerExecTime += runtime;
      // LTwoDotLaplaceInnerAll[((int)CounterLTwoDotLaplaceInner) % MOD] += runtime;
    }
  }
}

#endif
