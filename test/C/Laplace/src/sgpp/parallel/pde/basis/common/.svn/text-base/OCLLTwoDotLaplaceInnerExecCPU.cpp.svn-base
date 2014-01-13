#ifndef USE_MPI
#include "OCLLTwoDotLaplaceInner.hpp"

namespace sg {
  namespace parallel {
    namespace oclpdekernels {
      cl_kernel LTwoDotLaplaceInnerCPUKernel[NUMDEVS];
#define CPULSIZE 8



      char * ReadSources(const char *fileName) {

	FILE *file = fopen(fileName, "rb");
	if (!file)
	  {
	    printf("ERROR: Failed to open file '%s'\n", fileName);
	    return NULL;
	  }

	if (fseek(file, 0, SEEK_END))
	  {
	    printf("ERROR: Failed to seek file '%s'\n", fileName);
	    fclose(file);
	    return NULL;
	  }

	long size = ftell(file);
	if (size == 0)
	  {
	    printf("ERROR: Failed to check position on file '%s'\n", fileName);
	    fclose(file);
	    return NULL;
	  }

	rewind(file);

	char *src = (char *)malloc(sizeof(char) * size + 1);
	if (!src)
	  {
	    printf("ERROR: Failed to allocate memory for file '%s'\n", fileName);
	    fclose(file);
	    return NULL;
	  }

	printf("Reading file '%s' (size %ld bytes)\n", fileName, size);
	size_t res = fread(src, 1, sizeof(char) * size, file);
	if (res != sizeof(char) * size)
	  {
	    printf("ERROR: Failed to read file '%s'\n", fileName);
	    fclose(file);
	    free(src);
	    return NULL;
	  }

	src[size] = '\0'; /* NULL terminated */
	fclose(file);

	return src;
      
      }

      std::string LTwoDotLaplaceInnerCPUHeader() {
	std::stringstream stream_program_src;
	stream_program_src << "__kernel void multKernel(__global  REAL* ptrLevel,			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrIndex, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrLevel_int, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrAlpha, 			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrLcl_q,			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrLambda,			"  << std::endl;
	stream_program_src << "			 	__global  REAL* ptrResult,			"  << std::endl;
	stream_program_src << "                        	REAL TimestepCoeff)		                "  << std::endl;
	stream_program_src << "{									"  << std::endl;
	return stream_program_src.str(); 
      }


      void CompileInnerKernel(int id, std::string kernel_src, const char *filename, cl_kernel* kernel) {

	cl_int err = CL_SUCCESS;
	const char* source2 = ReadSources(filename);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source2,NULL, &err);
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
	    std::cout << "OCL Error: OpenCL Build Error. Error Code: " << err << std::endl;

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
	free((void *)source2);
      } 

      void CompileLTwoDotLaplaceInnerCPUKernels() {
	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    CompileInnerKernel(i, "multKernel", "MultKernelCPU.cl", LTwoDotLaplaceInnerCPUKernel);
	  }
      }
      void SetArgumentsLTwoDotLaplaceInnerCPU() {

	cl_int ciErrNum = CL_SUCCESS;
	int counter = 0;

	for(unsigned int i=0; i < num_devices; ++i) 
	  {
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevelInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrIndexInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLevel_intInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrAlphaInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrResultInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLcl_qInner[i]);
	    ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], counter++, sizeof(cl_mem), (void *) &d_ptrLambdaInner[i]);
	    oclCheckErr(ciErrNum, "clSetCPUKernelArg1 CPUKernel Construct");

	    counter = 0;
	  }
      }


      void ExecLTwoDotLaplaceInnerCPU(REAL * ptrAlpha,
				      REAL * ptrResult,
				      REAL * lcl_q,
				      REAL * lcl_q_inv,
				      REAL * ptrLevel,
				      REAL * ptrIndex,
				      REAL * ptrLevel_int,
				      size_t argStorageSize,
				      size_t argStorageDim) {

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
	  // size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * CPULSIZE);
	  multglobalworksize[0] = storageSizePadded;// std::min(storageSizePadded,storageSizePaddedStep);
	  multglobalworksize[1] = storageSizePadded/CPULSIZE;
	  size_t multglobal = storageSizePadded;


	  for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
	    multglobalworksize[0] =  std::min(multglobalworksize[0], multglobal-overallMultOffset);
	    size_t constantglobalworksize[2];
	    size_t constantlocalworksize[2];
	    constantglobalworksize[0] = multglobalworksize[0];
	    constantglobalworksize[1] = 1;
	    constantlocalworksize[0] = CPULSIZE;
	    constantlocalworksize[1] = 1;
	    for(unsigned int i = 0; i < num_devices; i++) {
	    // std::cout << " multglobalworksize[] " << multglobalworksize[0] << std::endl;
	      ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerCPUKernel[i], 7, sizeof(REAL), (void *) &TimestepCoeff);
	      oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	      ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
						LTwoDotLaplaceInnerCPUKernel[i], 
						2, 0, 
						constantglobalworksize, 
						constantlocalworksize,
						0, NULL, &GPUExecution[i]);
	      oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	    }

	  }
	  // std::cout << " END[] " << multglobalworksize[0] << std::endl;

	}

	ciErrNum |= clEnqueueReadBuffer(command_queue[0], d_ptrResultInner[0], CL_TRUE, 0,
					storageSize*sizeof(REAL), ptrResultPinnedInner, 0, NULL, NULL);
	oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL161");
	for( size_t i = 0; i < storageSize; i++) {
	  ptrResult[i] = ptrResultPinnedInner[i];
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
    void OCLPDEKernels::RunOCLKernelLTwoDotLaplaceInnerCPU(sg::base::DataVector& alpha,
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
      myStopwatch->start();
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
	CompileLTwoDotLaplaceInnerCPUKernels(); 
	SetArgumentsLTwoDotLaplaceInnerCPU();
	isVeryFirstTime = 0;
	isFirstTimeLTwoDotLaplaceInner = 0;
      }


      LTwoDotLaplaceInnerStartupTime += myStopwatch->stop();

      myStopwatch->start();
      ExecLTwoDotLaplaceInnerCPU(alpha.getPointer(), 
				 result.getPointer(), 
				 lcl_q, lcl_q_inv,
				 ptrLevel, ptrIndex, 
				 ptrLevel_int, argStorageSize, 
				 argStorageDim);
      double runtime = myStopwatch->stop();
      LTwoDotLaplaceInnerExecTime += runtime;
    }
  }
}

#endif
