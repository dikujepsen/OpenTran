/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "base/grid/type/LinearGrid.hpp"
#include "base/grid/generation/GridGenerator.hpp"
#include "parallel/datadriven/tools/DMVectorizationPaddingAssistant.hpp"

#include "parallel/pde/basis/linear/noboundary/operation/OperationLTwoDotProductOCLLinear.hpp"

#include <cmath>

#define GPU1 0


namespace sg {
  namespace parallel {

    OperationLTwoDotProductOCLLinear::OperationLTwoDotProductOCLLinear(sg::base::GridStorage* storage) : storage(storage) {

      this->lcl_q = new double[this->storage->dim()];
      this->OCLPDEKernelsHandle = OCLPDEKernels();

      this->level_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->level_int_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->index_ = new sg::base::DataMatrix(storage->size(), storage->dim());

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));

    }

    OperationLTwoDotProductOCLLinear::~OperationLTwoDotProductOCLLinear() {
      delete this->level_;
      delete this->level_int_;
      delete this->index_;
      delete[] lcl_q;
      #if GPU1
      this->OCLPDEKernelsHandle.
	CleanUpGPU();
      #endif
    }

    double OperationLTwoDotProductOCLLinear::l2dot(size_t i, size_t j, size_t dim) {
      double lid = level_->get(i, dim);
      double ljd = level_->get(j, dim);
      double iid = index_->get(i, dim);
      double ijd = index_->get(j, dim);
      double in_lid = level_int_->get(i, dim);
      double in_ljd = level_int_->get(j, dim);

      double res_one = (2.0 / 3.0) * in_lid * (iid == ijd);

      bool selector = (lid > ljd);
      double i1d = iid * (selector) + ijd * (!selector);
      double in_l1d = in_lid * (selector) + in_ljd * (!selector);
      double i2d = ijd * (selector) + iid * (!selector);
      double l2d = ljd * (selector) + lid * (!selector);
      double in_l2d = in_ljd * (selector) + in_lid * (!selector);

      double q = (i1d - 1) * in_l1d;
      double p = (i1d + 1) * in_l1d;
      bool overlap = (std::max(q, (i2d - 1) * in_l2d) < std::min(p, (i2d + 1) * in_l2d));

      double temp_res = 2.0 - fabs(l2d * q - i2d) - fabs(l2d * p - i2d);
      temp_res *= (0.5 * in_l1d);

      double res_two = temp_res * overlap; // Now mask result

      return (res_one * (lid == ljd) + res_two * (lid != ljd))  * lcl_q[dim];
    }

    void OperationLTwoDotProductOCLLinear::mult(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      result.setAll(0.0);
      for (size_t d = 0; d < this->storage->dim(); d++) {
        sg::base::BoundingBox* boundingBox = this->storage->getBoundingBox();
        this->lcl_q[d] = boundingBox->getIntervalWidth(d);
      }
      
#if GPU1
      
      this->OCLPDEKernelsHandle.
	RunOCLKernelLTwoDotInner(alpha, result,
				 this->lcl_q,
				 this->level_->getPointer(),
				 this->index_->getPointer(),
				 this->level_int_->getPointer(),
				 storage->size(),
				 storage->dim(),
				 storage);



      
#else

#pragma omp parallel
      {
#pragma omp for
        for (size_t ii = 0; ii < this->storage->size(); ii++) {
          for (size_t jj = 0; jj < this->storage->size(); jj++) {

	    double element = alpha[jj];

	    for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {
	      element *= l2dot(ii, jj, d_inner);
	    }

	    result[ii] += element;

          }
        }
      }
#endif
    }
    

  }
}
