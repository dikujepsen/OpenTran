/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "base/grid/type/LinearGrid.hpp"
#include "base/grid/generation/GridGenerator.hpp"
#include "parallel/datadriven/tools/DMVectorizationPaddingAssistant.hpp"

#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedOCLLinear.hpp"

#include <cmath>
#include <set>
#define GPU1 1

namespace sg {
  namespace parallel {

    OperationLaplaceVectorizedOCLLinear::OperationLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage, sg::base::DataVector& lambda) : storage(storage){


      this->lambda = new sg::base::DataVector(lambda);
      this->OCLPDEKernelsHandle = OCLPDEKernels();
      this->level_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->level_int_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->index_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      lcl_q = new double[this->storage->dim()];
      lcl_q_inv = new double[this->storage->dim()];

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));



    }

    OperationLaplaceVectorizedOCLLinear::OperationLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage) : storage(storage) {

      
      this->lambda = new base::DataVector(storage->dim());
      this->lambda->setAll(1.0);
      this->OCLPDEKernelsHandle = OCLPDEKernels();
      this->level_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->level_int_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->index_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      lcl_q = new double[this->storage->dim()];
      lcl_q_inv = new double[this->storage->dim()];

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));


    }


    OperationLaplaceVectorizedOCLLinear::~OperationLaplaceVectorizedOCLLinear() {
      delete this->level_;
      delete this->level_int_;
      delete this->index_;
      delete[] lcl_q;
      delete[] lcl_q_inv;
#if GPU1
      this->OCLPDEKernelsHandle.
	CleanUpGPU();
#endif
    }

    double OperationLaplaceVectorizedOCLLinear::gradient(
							 double i_level_grad,
							 double i_index_grad,
							 double j_level_grad,
							 double j_index_grad
							 ) {
      double grad;


      //only affects the diagonal of the stiffness matrix
      bool doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
      grad = i_level_grad * 2.0 * static_cast<double>(static_cast<int>(doGrad));

      return (grad);
    }

    double OperationLaplaceVectorizedOCLLinear::l2dot(size_t i, size_t j, size_t dim) {
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

      return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q[dim];
    }

    void OperationLaplaceVectorizedOCLLinear::mult(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      result.setAll(0.0);

      // fill q array
      for (size_t d = 0; d < this->storage->dim(); d++) {
        sg::base::BoundingBox* boundingBox = this->storage->getBoundingBox();
        lcl_q[d] = boundingBox->getIntervalWidth(d);
        lcl_q_inv[d] = 1.0 / boundingBox->getIntervalWidth(d);
      }

#if GPU1

      
      this->OCLPDEKernelsHandle.
      	RunOCLKernelLaplaceInner(alpha,result, lcl_q, lcl_q_inv,
					  this->level_->getPointer(),
					  this->index_->getPointer(),
					  this->level_int_->getPointer(),
					  lambda->getPointer(),
					  storage->size(),
					  storage->dim(),
					  storage);


#elif 1


#pragma omp parallel
      {
        double* gradient_temp = new double[this->storage->dim()];
        double* dot_temp = new double[this->storage->dim()];
#pragma omp for

        for (size_t ii = 0; ii < this->storage->size(); ii++) {
          for (size_t jj = 0; jj < this->storage->size(); jj++) {

            for (size_t d = 0; d < this->storage->dim(); d++) {

	      double level_i = level_->get(ii, d);
	      double index_i = index_->get(ii, d);
	      double level_j = level_->get(jj, d);
	      double index_j = index_->get(jj, d);
	      gradient_temp[d] = gradient(level_i,index_i,
					  level_j,index_j) * lcl_q_inv[d];
	      dot_temp[d] = l2dot(ii, jj, d);
	    }

	    for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
	      double element = alpha[jj];

	      for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {
		element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
	      }

	      result[ii] += lambda->get(d_outer) * element;
	    }
            
	  }
	}
	delete [] gradient_temp;
	delete [] dot_temp;

      }
#endif
    }

  }

}
