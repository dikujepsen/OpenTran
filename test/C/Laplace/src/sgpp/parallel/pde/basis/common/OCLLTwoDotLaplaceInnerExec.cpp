#ifndef USE_MPI
#include "OCLLTwoDotLaplaceInner.hpp"

namespace sg {
  namespace parallel {
    namespace oclpdekernels {
      size_t * GlobalSizeN_ii;
      size_t * GlobalOffsetN_ii;


      void SetGlobalSizeNInner() {
	size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * LSIZE);
	std::cout << " storageSizePaddedStep " << storageSizePaddedStep << std::endl;
	size_t minStepSize = std::min(storageSizePadded,storageSizePaddedStep);
	size_t symResultY_size = std::min(constant_buffer_iterations_noboundary,minStepSize);
	size_t acc = 0;
	size_t count = 0;
	while (acc < storageSizePadded) {
	  count++;
	  acc += symResultY_size;
	}
	std::cout << " count " << count << " symResultY_size " << symResultY_size << std::endl;
	GlobalSizeN_ii = (size_t*)calloc(count, sizeof(size_t));
	GlobalOffsetN_ii = (size_t*)calloc(count, sizeof(size_t));
	
	acc = storageSizePadded;
	size_t count2 = 0;
	size_t offset = 0;
	while (0 < acc) {
	  std::cout << " acc " << acc << std::endl;
	  GlobalSizeN_ii[count2] = acc;
	  GlobalOffsetN_ii[count2++] = offset;
	  acc -= std::min(symResultY_size, acc);
	  offset += std::min(symResultY_size, storageSizePadded-offset);
	}
	for (size_t i = 0; i < count; i++) {
	  GlobalSizeN_ii[i] /= LSIZE;
	  GlobalOffsetN_ii[i] /= LSIZE;
	  std::cout << " GlobalSizeN_ii["<<i<<"] = "<< GlobalSizeN_ii[i] << std::endl;
	  std::cout << " GlobalOffsetN_ii["<<i<<"] = "<< GlobalOffsetN_ii[i] << std::endl;
	}
      }

      void ExecLTwoDotLaplaceInner(REAL * ptrAlpha,
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
	  size_t storageSizePaddedStep = std::min(storageSizePadded  / num_devices, par_result_max_size / (storageSizePadded) * LSIZE);
	  multglobalworksize[0] = std::min(storageSizePadded,storageSizePaddedStep);
	  multglobalworksize[1] = storageSizePadded/LSIZE;
	  size_t multglobal = storageSizePadded / num_devices;
	  // 	  std::cout << " multglobal " << multglobal << std::endl;
	  for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
// 	    myStopwatch->start();
	    multglobalworksize[0] =  std::min(multglobalworksize[0], multglobal-overallMultOffset);

	    for(unsigned int i = 0; i < num_devices; i++) {
	      size_t overallMultOffset2 = overallMultOffset + i*multglobal;
	      ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerKernel[i], 8, sizeof(cl_ulong), (void *) &overallMultOffset2);
	    }
	    oclCheckErr(ciErrNum, "clSetKernelArgL246");
	    size_t constantglobalworksize[2];
	    size_t constantlocalworksize[2];
	    size_t constantglobal = multglobalworksize[0];
	    constantglobalworksize[0] = std::min(constant_buffer_iterations_noboundary,constantglobal);

	    constantglobalworksize[1] = multglobalworksize[1];
	    constantlocalworksize[0] = LSIZE;
	    constantlocalworksize[1] = 1;
	    // 	    std::cout << " NEW " << std::endl;
	    for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	      constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);

	      // 	      std::cout << " constantglobalworksize[0] " << constantglobalworksize[0] << std::endl;
	      // 	      std::cout << " constantglobalworksize[1] " << constantglobalworksize[1] << std::endl;
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
						 d_ptrLevelIndexLevelintconInner[i], 
						 CL_FALSE, 0 ,
						 constantglobalworksize[0]*3*dims*sizeof(REAL), 
						 ptrLevelIndexLevelintInner+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
						 1,
						 &GPUExecution[i] , &GPUDone[i]);
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL306");
		size_t jj = (ConstantMemoryOffset) / LSIZE;

		ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerKernel[i], 9, sizeof(cl_ulong), (void *) &jj);
		ciErrNum |= clSetKernelArg(LTwoDotLaplaceInnerKernel[i], 10, sizeof(REAL), (void *) &TimestepCoeff);
		// 		constantglobalworksize[1] = GlobalSizeN_ii[kernelcounter++];
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL302");
		ciErrNum = clEnqueueNDRangeKernel(command_queue[i], 
						  LTwoDotLaplaceInnerKernel[i], 
						  2, 0, 
						  constantglobalworksize, 
						  constantlocalworksize,
						  1, &GPUDone[i], &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel2314");
	      }
// 	      for(unsigned int i = 0; i < num_devices; i++) {
// 		ciErrNum |= clFinish(command_queue[i]);
// 		oclCheckErr(ciErrNum, "clFinishLapIL105");
// 	      }
// 	      LTwoDotLaplaceInnerProfilingAcc += AccumulateTiming(GPUExecution, 0);
	    }

	    // 	      LTwoDotLaplaceInnerProfilingWait += AccumulateWaiting(GPUExecution, 0);
	    

	    //REDUCE
	    size_t overallReduceOffset = 0;//overallMultOffset*LSIZE; 
	    for(unsigned int i = 0; i < num_devices; i++) {

	      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	      oclCheckErr(ciErrNum, "clSetKernelArgLapIL2199");
	      size_t newnum_groups = multglobalworksize[0] / LSIZE ;
	      ciErrNum |= clSetKernelArg(ReduceInnerKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
	      oclCheckErr(ciErrNum, "clSetKernelArgLapIL340");

	      size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	      size_t local2[] = {LSIZE,1};
	      ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
						 ReduceInnerKernel[i], 
						 2, 0, reduceglobalworksize2, 
						 local2,
						 0, NULL, &GPUExecution[i]);
	      oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
	    }
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
					      ptrResultPinnedInner + i * storageSizePadded, 
					      1, &GPUExecution[i], &GPUDone[i]);

	    }
	  oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
	  ciErrNum |= clWaitForEvents(num_devices, GPUDone);
	  oclCheckErr(ciErrNum, "clWaitForEventsLapIL2147");

	  for (size_t j = 0; j < num_devices; j++) {
	    for (size_t i = 0; i < storageSize; i++) {
	      ptrResult[i] += ptrResultPinnedInner[j*storageSizePadded + i];
	    }
	  }

	} else {
	  for(unsigned int i = 0;i < num_devices; i++) {    
	    ciErrNum |= clEnqueueReadBuffer(command_queue[i], 
					    d_ptrResultInner[i], CL_TRUE, 0,
					    storageSize*sizeof(REAL), 
					    ptrResultPinnedInner, 
					    0, NULL, NULL);
	    oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL161");
	  }
	  // 	  clWaitForEvents(num_devices, GPUDone);
	  for( size_t i = 0; i < storageSize; i++) {
	    ptrResult[i] = ptrResultPinnedInner[i];
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
    void OCLPDEKernels::RunOCLKernelLTwoDotLaplaceInner(sg::base::DataVector& alpha,
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
	SetGlobalSizeNInner();
	CompileLTwoDotLaplaceInnerKernels(); 
	SetArgumentsLTwoDotLaplaceInner();
	isVeryFirstTime = 0;
	isFirstTimeLTwoDotLaplaceInner = 0;
      }


      LTwoDotLaplaceInnerStartupTime += myStopwatch->stop();

      myStopwatch->start();
      ExecLTwoDotLaplaceInner(alpha.getPointer(), 
			      result.getPointer(), 
			      lcl_q, lcl_q_inv,
			      ptrLevel, ptrIndex, 
			      ptrLevel_int, argStorageSize, 
			      argStorageDim);
      double runtime = myStopwatch->stop();
      LTwoDotLaplaceInnerExecTime += runtime;
      // LTwoDotLaplaceInnerAll[((int)CounterLTwoDotLaplaceInner) % MOD] += runtime;
    }
  }
}

#endif
