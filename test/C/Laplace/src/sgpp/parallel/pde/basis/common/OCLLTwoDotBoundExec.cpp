#ifndef USE_MPI
#include "OCLLTwoDotBound.hpp"

namespace sg {
  namespace parallel {
    namespace oclpdekernels {

      void ExecLTwoDotBound(REAL * ptrAlpha,
			    REAL * ptrResult,
			    REAL * lcl_q) {

	cl_int ciErrNum = CL_SUCCESS;
	cl_event GPUDone[NUMDEVS];
	cl_event GPUDoneLcl[NUMDEVS];
	cl_event GPUExecution[NUMDEVS];
  
	for (size_t d_outer = 0; d_outer < dims ; d_outer++) {
	  ptrLcl_qBound[d_outer] = lcl_q[d_outer];
	}


	for(size_t i = 0; i < num_devices; i++) {

	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, 0,
					   storageSizeBound*sizeof(REAL), ptrAlpha, 0, 0, &GPUDone[i]);
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrLcl_qBound[i], CL_FALSE, 0,
					   lcl_q_sizeBound*sizeof(REAL), ptrLcl_qBound, 0, 0, &GPUDoneLcl[i]);
	}
	clWaitForEvents(num_devices, GPUDone);
	clWaitForEvents(num_devices, GPUDoneLcl);
	oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");

	for(size_t i = 0; i < num_devices; i++) {
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrResultBound[i], 
					   CL_FALSE, 0,
					   Inner_result_sizeBound*sizeof(REAL), 
					   ptrResultZeroBound, 0, 0, &GPUDone[i]);
	  ciErrNum |= clEnqueueWriteBuffer(command_queue[i], d_ptrAlphaBound[i], CL_FALSE, storageSizeBound*sizeof(REAL),
					   alphaend_sizeBound*sizeof(REAL), ptrAlphaEndBound, 0, 0, &GPUExecution[i]);
	}
	oclCheckErr(ciErrNum, "clEnqueueWriteBuffer mult");  
	clWaitForEvents(num_devices, GPUDone);

	{
	  size_t multglobalworksize[2];
	  size_t storageSizePaddedStep = std::min(storageSizePaddedBound / num_devices, Inner_par_result_max_sizeBound / (storageSizePaddedBound) * LSIZE);
	  multglobalworksize[0] = std::min(storageSizePaddedBound,storageSizePaddedStep);
	  multglobalworksize[1] = storageInnerSizePaddedBound/ LSIZE;

	  size_t multglobal = storageSizePaddedBound / num_devices;

	  for(size_t overallMultOffset = 0; overallMultOffset < multglobal; overallMultOffset+= std::min(multglobalworksize[0], multglobal - overallMultOffset)) {
	    multglobalworksize[0] = std::min(multglobalworksize[0], multglobal-overallMultOffset);

	    for(unsigned int i = 0; i < num_devices; i++) {
	      size_t overallMultOffset2 = overallMultOffset +i*multglobal;
	      ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], 7, sizeof(cl_ulong), (void *) &overallMultOffset2);
	      oclCheckErr(ciErrNum, "clSetKernelArgL1660");
	    }
	  
	    size_t constantglobalworksize[2];
	    size_t constantlocalworksize[2];
	    size_t constantglobal = multglobalworksize[0];
	    constantglobalworksize[0] = std::min(constant_buffer_iterations,constantglobal);
	    constantglobalworksize[1] = multglobalworksize[1]; // already LSIZE pad
	    constantlocalworksize[0] = LSIZE;
	    constantlocalworksize[1] = 1;
	    for(size_t ConstantMemoryOffset = 0; ConstantMemoryOffset < constantglobal; ConstantMemoryOffset+= std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset)) {
	      constantglobalworksize[0] = std::min(constantglobalworksize[0],constantglobal-ConstantMemoryOffset);
	      for(unsigned int i = 0; i < num_devices; i++) {
		ciErrNum |= clEnqueueWriteBuffer(command_queue[i], 
						 d_ptrLevelIndexLevelintconBound[i], 
						 CL_FALSE, 0 ,
						 constantglobalworksize[0]*3*dims*sizeof(REAL), 
						 ptrLevelIndexLevelintBound+(overallMultOffset + i*multglobal + ConstantMemoryOffset)*3*dims, 
						 1, 
						 &GPUExecution[i], &GPUDone[i]);
		oclCheckErr(ciErrNum, "clEnqueueWriteBufferL2157");
		size_t jj = (ConstantMemoryOffset) / LSIZE;
		ciErrNum |= clSetKernelArg(LTwoDotBoundKernel[i], 8, sizeof(cl_ulong), (void *) &jj);
		oclCheckErr(ciErrNum, "clSetKernelArgL2150");
		ciErrNum = clEnqueueNDRangeKernel(command_queue[i], LTwoDotBoundKernel[i], 2, 0, constantglobalworksize, constantlocalworksize,
						  1, &GPUDone[i],
						  &GPUExecution[i]);
		oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel3115");

	      }
	    }

	    size_t overallReduceOffset = 0;
	    for(unsigned int i = 0; i < num_devices; i++) {
	      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 2, sizeof(cl_ulong), (void *) &overallReduceOffset);
	      size_t newnum_groups = multglobalworksize[0] / LSIZE;
	      ciErrNum |= clSetKernelArg(ReduceBoundKernel[i], 3, sizeof(cl_ulong), (void *) &newnum_groups);
	      oclCheckErr(ciErrNum, "clSetKernelArgL1205");
      
	      size_t reduceglobalworksize2[] = {multglobalworksize[1]*LSIZE, 1};
	      size_t local2[] = {LSIZE,1};

	      ciErrNum |= clEnqueueNDRangeKernel(command_queue[i], 
						 ReduceBoundKernel[i], 2, 0, 
						 reduceglobalworksize2, local2,
						 0, NULL, &GPUExecution[i]);
	      oclCheckErr(ciErrNum, "clEnqueueNDRangeKernel1213");
	    }
	    // 	    for(unsigned int i = 0; i < num_devices; i++) {
	    // 	      ciErrNum = clFinish(command_queue[i]);
	    // 	      oclCheckErr(ciErrNum, "clFinish");
      
	    // 	    }
	  }
	}


	{
	  if (num_devices > 1) {
  
	    for(unsigned int i = 0;i < num_devices; i++) 
	      {    
		ciErrNum |= clEnqueueReadBuffer(command_queue[i], 
						d_ptrResultBound[i], 
						CL_FALSE, 0,
						storageInnerSizeBound*sizeof(REAL), 
						ptrResultPinnedBound + i * storageInnerSizePaddedBound, 
						1, &GPUExecution[i], &GPUDone[i]);

	      }
	    oclCheckErr(ciErrNum, "clEnqueueReadBufferLapIL2145");
	    ciErrNum |= clWaitForEvents(num_devices, GPUDone);
	    oclCheckErr(ciErrNum, "clWaitForEventsLapIL2147");

	    for (size_t i = 0; i < storageInnerSizeBound; i++) {
	      ptrResultBound[i] = 0.0;

	    }
	     
	    for (size_t j = 0; j < num_devices; j++) {
	      for (size_t i = 0; i < storageInnerSizeBound; i++) {
		ptrResultBound[i] += ptrResultPinnedBound[j*storageInnerSizePaddedBound + i];

	      }
	    }
	  } else {
	    for(unsigned int i = 0;i < num_devices; i++) 
	      {    
		clEnqueueReadBuffer(command_queue[i], d_ptrResultBound[i], 
				    CL_FALSE, 0,
				    storageInnerSizeBound*sizeof(REAL), 
				    ptrResultPinnedBound, 1, &GPUExecution[i], &GPUDone[i]);
	      }
	    clWaitForEvents(num_devices, GPUDone);
	    for (size_t j = 0; j < storageInnerSizeBound; j++) {
	      ptrResultBound[j] = ptrResultPinnedBound[j];
	    }    
	  }
	  for (unsigned i = 0; i < storageInnerSizeBound; i++) {
	    ptrResult[offsetBound[i]] = ptrResultBound[i];
	  }
	}
#if TOTALTIMING
	CounterLTwoDotBound += 1.0;
#endif	

	for(size_t i = 0; i < num_devices; i++) {
	  clReleaseEvent(GPUExecution[i]);
	  clReleaseEvent(GPUDone[i]);
	  clReleaseEvent(GPUDoneLcl[i]);
	}  
      }

    }
    using namespace oclpdekernels;

    void OCLPDEKernels::RunOCLKernelLTwoDotBound(sg::base::DataVector& alpha,
						 sg::base::DataVector& result,
						 REAL * lcl_q,
						 REAL * ptrLevel,
						 REAL * ptrIndex,
						 REAL * ptrLevel_int,
						 size_t argStorageSize,
						 size_t argStorageDim,
						 sg::base::GridStorage* storage) {
      if (isVeryFirstTime) {
	StartUpGPU();
      } 
      if (isFirstTimeLaplaceBound && 
	  isFirstTimeLTwoDotBound &&
	  isFirstTimeLTwoDotLaplaceBound) {
	SetBuffersBound(ptrLevel,
			ptrIndex,
			ptrLevel_int,
			argStorageSize,
			argStorageDim,storage);
      }
      if (isFirstTimeLTwoDotBound) {
	CompileLTwoDotBoundKernels(); 
	SetArgumentsLTwoDotBound();
	isVeryFirstTime = 0; 
	isFirstTimeLTwoDotBound = 0;
      }
      myStopwatch->start();
      ExecLTwoDotBound(alpha.getPointer(), result.getPointer(),lcl_q);
      LTwoDotBoundExecTime += myStopwatch->stop();
    }
        
  }
}
#endif
