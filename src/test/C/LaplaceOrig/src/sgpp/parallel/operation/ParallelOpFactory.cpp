/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Valeriy Khakhutskyy (khakhutv@in.tum.de), Dirk Pflueger (pflueged@in.tum.de)
// @author Alexander Heinecke (alexander.heinecke@mytum.de)
// @author Roman Karlstetter (karlstetter@mytum.de)

#ifdef USEMIC
#include "parallel/datadriven/basis/common/mic/MICKernel.hpp"
#include "parallel/datadriven/basis/common/mic/MICCPUHybridKernel.hpp"
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif
#include "parallel/datadriven/basis/linear/noboundary/operation/impl/MICLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/MICModLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/MICModLinearMask.hpp"
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif
#endif


#include "parallel/operation/ParallelOpFactory.hpp"


#include "base/exception/factory_exception.hpp"

#include "parallel/datadriven/operation/OperationMultipleEvalIterative.hpp"
#include "parallel/datadriven/basis/linear/noboundary/operation/impl/X86SimdLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/X86SimdModLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/X86SimdModLinearMask.hpp"

#include "parallel/datadriven/basis/common/CPUKernel.hpp"
#ifdef USEOCL
#include "parallel/datadriven/basis/common/ocl/OCLKernel.hpp"
#include "parallel/datadriven/basis/common/ocl/OCLKernelImpl.hpp"
#include "parallel/datadriven/basis/common/ocl/OCLCPUHybridKernel.hpp"
#include "parallel/datadriven/basis/linear/noboundary/operation/impl/OCLLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/OCLModLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/impl/OCLModLinearMask.hpp"
#endif

#ifdef USEARBB
#include "parallel/datadriven/basis/linear/noboundary/operation/OperationMultipleEvalIterativeArBBLinear.hpp"
#endif

#ifdef USEOCL
#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedOCLLinear.hpp"
#include "parallel/pde/basis/linear/boundary/operation/OperationLaplaceVectorizedOCLLinearBoundary.hpp"
#include "parallel/pde/basis/linear/noboundary/operation/OperationLTwoDotProductOCLLinear.hpp"
#include "parallel/pde/basis/linear/boundary/operation/OperationLTwoDotProductOCLLinearBoundary.hpp"
//  LU and combined operator
#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedLinearLU.hpp"
#include "parallel/pde/basis/linear/noboundary/operation/OperationLTwoDotLaplaceVectorizedOCLLinear.hpp"
#include "parallel/pde/basis/linear/boundary/operation/OperationLTwoDotLaplaceVectorizedOCLLinearBoundary.hpp"

#endif
#ifdef USEMIC
#include "parallel/datadriven/basis/linear/noboundary/operation/OperationMultipleEvalIterativeMICLinear.hpp"
#include "parallel/datadriven/basis/linear/noboundary/operation/OperationMultipleEvalIterativeHybridX86SimdMICLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/OperationMultipleEvalIterativeMICModLinear.hpp"
#include "parallel/datadriven/basis/modlinear/operation/OperationMultipleEvalIterativeHybridX86SimdMICModLinear.hpp"
#endif



#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedLinear.hpp"
#include "parallel/pde/basis/linear/boundary/operation/OperationLaplaceVectorizedLinearBoundary.hpp"

#include <cstring>


namespace sg {

  namespace op_factory {

    parallel::OperationMultipleEvalVectorized* createOperationMultipleEvalVectorized(base::Grid& grid, const parallel::VectorizationType& vecType, base::DataMatrix* dataset,
        size_t gridFrom, size_t gridTo, size_t datasetFrom, size_t datasetTo) {
      // handle default upper boundaries
      if (gridTo == std::numeric_limits<size_t>::max()) {
        gridTo = grid.getStorage()->size();
      }

      if (datasetTo == std::numeric_limits<size_t>::max()) {
        datasetTo = dataset->getNcols();
      }

      // get env var
      const char* modlinear_mode = getenv("SGPP_MODLINEAR_EVAL");

      if (modlinear_mode == NULL) {
        modlinear_mode = "mask";
      }

      if (strcmp(grid.getType(), "linear") == 0) {
        if (vecType == parallel::X86SIMD) {
          return new parallel::OperationMultipleEvalIterative<parallel::CPUKernel<parallel::X86SimdLinear> >(grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#ifdef USEOCL
        else if (vecType == parallel::OpenCL) {
          return new parallel::OperationMultipleEvalIterative <
                 parallel::OCLKernel<parallel::OCLLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        } else if (vecType == parallel::Hybrid_X86SIMD_OpenCL) {
          return new parallel::OperationMultipleEvalIterative < parallel::OCLCPUHybridKernel < parallel::X86SimdLinear,
                 parallel::OCLLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#endif
#ifdef USEARBB
        else if (vecType == parallel::ArBB) {
          return new parallel::OperationMultipleEvalIterativeArBBLinear(grid.getStorage(), dataset);
        }

#endif
#ifdef USEMIC
        else if (vecType == parallel::MIC) {
          return new parallel::OperationMultipleEvalIterative<parallel::MICKernel<parallel::MICLinear> > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#ifdef __INTEL_OFFLOAD // Hybrid CPU MIC Mode only makes sense in offload mode
        else if (vecType == parallel::Hybrid_X86SIMD_MIC) {
          return new parallel::OperationMultipleEvalIterative<parallel::MICCPUHybridKernel<parallel::X86SimdLinear, parallel::MICLinear> > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#endif
#endif
        else {
          throw base::factory_exception("Unsupported vectorization type");
        }
      }

      else if (strcmp(grid.getType(), "linearBoundary") == 0
               || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {
        if (vecType == parallel::X86SIMD) {
          return new parallel::OperationMultipleEvalIterative<parallel::CPUKernel<parallel::X86SimdLinear> >(
										       grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#ifdef USEOCL
        else if (vecType == parallel::OpenCL) {
          return new parallel::OperationMultipleEvalIterative <
                 parallel::OCLKernel < parallel::OCLLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        } else if (vecType == parallel::Hybrid_X86SIMD_OpenCL) {
          return new parallel::OperationMultipleEvalIterative <
                 parallel::OCLCPUHybridKernel < parallel::X86SimdLinear,
                 parallel::OCLLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#endif
#ifdef USEARBB
        else if (vecType == parallel::ArBB) {
          return new parallel::OperationMultipleEvalIterativeArBBLinear(grid.getStorage(), dataset);
        }

#endif
#ifdef USEMIC
        else if (vecType == parallel::MIC) {
          return new parallel::OperationMultipleEvalIterative<parallel::MICKernel<parallel::MICLinear> >
                 (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#ifdef __INTEL_OFFLOAD // Hybrid CPU MIC Mode only makes sense in offload mode
        else if (vecType == parallel::Hybrid_X86SIMD_MIC) {
          return new parallel::OperationMultipleEvalIterative<parallel::MICCPUHybridKernel<parallel::X86SimdLinear, parallel::MICLinear> >
                 (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
        }

#endif
#endif
        else {
          throw base::factory_exception("Unsupported vectorization type");
        }
      } else if (strcmp(grid.getType(), "modlinear") == 0) {
        if (vecType == parallel::X86SIMD) {
          if (strcmp(modlinear_mode, "orig") == 0) {
            return new parallel::OperationMultipleEvalIterative<parallel::CPUKernel<parallel::X86SimdModLinear> >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else if (strcmp(modlinear_mode, "mask") == 0) {
            return new parallel::OperationMultipleEvalIterative<parallel::CPUKernel<parallel::X86SimdModLinearMask> >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else {
            throw base::factory_exception("ParallelOpFactory: SGPP_MODLINEAR_EVAL must be 'mask' or 'orig'.");
          }
        }

#ifdef USEOCL
        else if (vecType == parallel::OpenCL) {
          if (strcmp(modlinear_mode, "orig") == 0) {
            return new parallel::OperationMultipleEvalIterative <
                   parallel::OCLKernel < parallel::OCLModLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else if (strcmp(modlinear_mode, "mask") == 0) {
            return new parallel::OperationMultipleEvalIterative <
                   parallel::OCLKernel < parallel::OCLModLinearMask<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else {
            throw base::factory_exception("ParallelOpFactory: SGPP_MODLINEAR_EVAL must be 'mask' or 'orig'.");
          }

        } else if (vecType == parallel::Hybrid_X86SIMD_OpenCL) {
          if (strcmp(modlinear_mode, "orig") == 0) {
            return new parallel::OperationMultipleEvalIterative <
                   parallel::OCLCPUHybridKernel < parallel::X86SimdModLinear,
                   parallel::OCLModLinear<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else if (strcmp(modlinear_mode, "mask") == 0) {
            return new parallel::OperationMultipleEvalIterative <
                   parallel::OCLCPUHybridKernel < parallel::X86SimdModLinearMask,
                   parallel::OCLModLinearMask<double> > > (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else {
            throw base::factory_exception("ParallelOpFactory: SGPP_MODLINEAR_EVAL must be 'mask' or 'orig'.");
          }
        }

#endif
#ifdef USEMIC
        else if (vecType == parallel::MIC) {
          if (strcmp(modlinear_mode, "orig") == 0) {
            return new parallel::OperationMultipleEvalIterative < parallel::MICKernel < parallel::MICModLinear > >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else if (strcmp(modlinear_mode, "mask") == 0) {
            return new parallel::OperationMultipleEvalIterative < parallel::MICKernel < parallel::MICModLinearMask > >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else {
            throw base::factory_exception("ParallelOpFactory: SGPP_MODLINEAR_EVAL must be 'mask' or 'orig'.");
          }
        }

#ifdef __INTEL_OFFLOAD // Hybrid CPU MIC Mode only makes sense in offload mode
        else if (vecType == parallel::Hybrid_X86SIMD_MIC) {
          if (strcmp(modlinear_mode, "orig") == 0) {
            return new parallel::OperationMultipleEvalIterative < parallel::MICCPUHybridKernel < parallel::X86SimdModLinear, parallel::MICModLinear > >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else if (strcmp(modlinear_mode, "mask") == 0) {
            return new parallel::OperationMultipleEvalIterative < parallel::MICCPUHybridKernel < parallel::X86SimdModLinearMask, parallel::MICModLinearMask > >
                   (grid.getStorage(), dataset, gridFrom, gridTo, datasetFrom, datasetTo);
          } else {
            throw base::factory_exception("ParallelOpFactory: SGPP_MODLINEAR_EVAL must be 'mask' or 'orig'.");
          }
        }

#endif
#endif
        else {
          throw base::factory_exception("Unsupported vectorization type");
        }
      } else {
        throw base::factory_exception("ParallelOpFactory: OperationMultipleEvalVectorized is not implemented for this grid type.");
      }
    }

#ifdef USEOCL
    base::OperationMatrix* createOperationLTwoDotProductVectorized(base::Grid& grid, const parallel::VectorizationType& vecType) {

      if (strcmp(grid.getType(), "linear") == 0) {
#ifdef USEOCL
        return new parallel::OperationLTwoDotProductOCLLinear(grid.getStorage());
#endif
      } else
	if (strcmp(grid.getType(), "linearBoundary") == 0 || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {
#ifdef USEOCL
	  return new parallel::OperationLTwoDotProductOCLLinearBoundary(grid.getStorage());
#endif
	} else
	  throw base::factory_exception("OperationLTwoDotProductVectorized is not implemented for this grid type.");
    }
#endif    

    base::OperationMatrix* createOperationLaplaceVectorized(base::Grid& grid, const parallel::VectorizationType& vecType) {
      if (strcmp(grid.getType(), "linear") == 0) {
#ifdef USEOCL
	if (vecType == parallel::OpenCL) {
	  return new parallel::OperationLaplaceVectorizedOCLLinear(grid.getStorage());
	} else {
#endif
	  return new parallel::OperationLaplaceVectorizedLinear(grid.getStorage());
#ifdef USEOCL
	}
#endif
      } else if (strcmp(grid.getType(), "linearBoundary") == 0 || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {
#ifdef USEOCL
	if (vecType == parallel::OpenCL) {
	  return new parallel::OperationLaplaceVectorizedOCLLinearBoundary(grid.getStorage());
	} else {
#endif

	  return new parallel::OperationLaplaceVectorizedLinearBoundary(grid.getStorage()); 
#ifdef USEOCL
	}
#endif

      } else {
        throw base::factory_exception("ParallelOpFactory: OperationLaplaceVectorized is not implemented for this grid type.");
      }

    }

#ifdef USEOCL
    base::OperationMatrix* createOperationLaplaceVectorizedLU(base::Grid& grid, const parallel::VectorizationType& vecType) {
      if (strcmp(grid.getType(), "linear") == 0) {
	return new parallel::OperationLaplaceVectorizedLinearLU(grid.getStorage());
      } else if (strcmp(grid.getType(), "linearBoundary") == 0 || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {

	return new parallel::OperationLaplaceVectorizedLinearBoundary(grid.getStorage()); 

      } else {
	throw base::factory_exception("OperationLaplaceVectorized is not implemented for this grid type.");
      }
    }

    base::OperationMatrix* createOperationLTwoDotLaplaceVectorized(base::Grid& grid,sg::base::DataVector& lambda, const parallel::VectorizationType& vecType) {
      if (strcmp(grid.getType(), "linear") == 0) {
	return new parallel::OperationLTwoDotLaplaceVectorizedOCLLinear(grid.getStorage(), lambda);
      } else if (strcmp(grid.getType(), "linearBoundary") == 0 || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {
	return new parallel::OperationLTwoDotLaplaceVectorizedOCLLinearBoundary(grid.getStorage(), lambda);
      }  else {
	throw base::factory_exception("OperationLTwoDotLaplaceVectorized is not implemented for this grid type.");
      }
    }


#endif

    base::OperationMatrix* createOperationLaplaceVectorized(base::Grid& grid, sg::base::DataVector& lambda, const parallel::VectorizationType& vecType) {
      if (strcmp(grid.getType(), "linear") == 0) {

#ifdef USEOCL
	if (vecType == parallel::OpenCL) {
	  return new parallel::OperationLaplaceVectorizedOCLLinear(grid.getStorage(), lambda);
	} else {
#endif
	  return new parallel::OperationLaplaceVectorizedLinear(grid.getStorage(), lambda);
#ifdef USEOCL
	}
#endif
      } else if (strcmp(grid.getType(), "linearBoundary") == 0 || strcmp(grid.getType(), "linearTrapezoidBoundary") == 0) {
#ifdef USEOCL
	if (vecType == parallel::OpenCL) {	
	  return new parallel::OperationLaplaceVectorizedOCLLinearBoundary(grid.getStorage(), lambda);
	} else {
#endif
	  return new parallel::OperationLaplaceVectorizedLinearBoundary(grid.getStorage(),lambda);
#ifdef USEOCL
	}
#endif

      } else {
	throw base::factory_exception("OperationLaplaceVectorized is not implemented for this grid type.");
      }
    }


  }
}
