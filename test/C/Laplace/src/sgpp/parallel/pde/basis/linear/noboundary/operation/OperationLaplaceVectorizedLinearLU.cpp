/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "base/grid/type/LinearGrid.hpp"
#include "base/grid/generation/GridGenerator.hpp"

#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedLinearLU.hpp"

#include <cmath>

namespace sg {
  namespace parallel {

    OperationLaplaceVectorizedLinearLU::OperationLaplaceVectorizedLinearLU(sg::base::GridStorage* storage) : storage(storage) {

      this->OCLPDEKernelsHandle = OCLPDEKernels();

      this->level_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->level_int_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->index_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      lcl_q = new double[this->storage->dim()];
      lcl_q_inv = new double[this->storage->dim()];

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));
      this->lambda_ = new sg::base::DataVector(storage->dim());
      this->lambda_->setAll(1.0);

    }

    OperationLaplaceVectorizedLinearLU::OperationLaplaceVectorizedLinearLU(sg::base::GridStorage* storage, sg::base::DataVector& lambda) : storage(storage) {

      this->OCLPDEKernelsHandle = OCLPDEKernels();

      this->level_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->level_int_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      this->index_ = new sg::base::DataMatrix(storage->size(), storage->dim());
      lcl_q = new double[this->storage->dim()];
      lcl_q_inv = new double[this->storage->dim()];

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));
      this->lambda_ = new sg::base::DataVector(lambda);
    }

    OperationLaplaceVectorizedLinearLU::~OperationLaplaceVectorizedLinearLU() {
      delete this->level_;
      delete this->level_int_;
      delete this->index_;
      delete[] lcl_q;
      delete[] lcl_q_inv;
      this->OCLPDEKernelsHandle.
	CleanUpGPU();

      if (this->lambda_ != NULL)
        delete this->lambda_;
    }

    double OperationLaplaceVectorizedLinearLU::gradient(size_t i, size_t j, size_t dim) {
      double grad;

      double i_level_grad = level_->get(i, dim);
      double i_index_grad = index_->get(i, dim);
      double j_level_grad = level_->get(j, dim);
      double j_index_grad = index_->get(j, dim);

      //only affects the diagonal of the stiffness matrix
      bool doGrad = ((i_level_grad == j_level_grad) && (i_index_grad == j_index_grad));
      grad = i_level_grad * 2.0 * static_cast<double>(static_cast<int>(doGrad));

      return (grad * lcl_q_inv[dim]);
    }

    double OperationLaplaceVectorizedLinearLU::l2dot(size_t i, size_t j, size_t dim) {
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

    void OperationLaplaceVectorizedLinearLU::mult(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      result.setAll(0.0);
      
      
      // fill q array
      for (size_t d = 0; d < this->storage->dim(); d++) {
        sg::base::BoundingBox* boundingBox = this->storage->getBoundingBox();
        lcl_q[d] = boundingBox->getIntervalWidth(d);
        lcl_q_inv[d] = 1.0 / boundingBox->getIntervalWidth(d);
      }

#if 1

      //        size_t gbl_count=0;

#pragma omp parallel
      {
        double* gradient_temp = new double[this->storage->dim()];
        double* dot_temp = new double[this->storage->dim()];
        double all_gradient_zero = 0;
        //  size_t lcl_count = 0;

#pragma omp for

        for (size_t ii = 0; ii < this->storage->size(); ii++) {
          for (size_t jj = 0; jj < this->storage->size(); jj++) {
            all_gradient_zero = 0;

            for (size_t d = 0; d < this->storage->dim(); d++) {
              gradient_temp[d] = gradient(ii, jj, d);
              all_gradient_zero += gradient_temp[d];
            }

            if (all_gradient_zero > 0) {
              for (size_t d = 0; d < this->storage->dim(); d++) {
                dot_temp[d] = l2dot(ii, jj, d);
              }

              for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
                double element = alpha[jj];
                for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {
                  element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (gradient_temp[d_inner] * (d_outer == d_inner)));
                }

                result[ii] += (this->lambda_->get(d_outer) * element);
              }

            }
          }
        }

        delete [] gradient_temp;
        delete [] dot_temp;

      }
#else
      //Original Version
#pragma omp parallel for

      for (size_t i = 0; i < this->storage->size(); i++) {
        for (size_t j = 0; j < this->storage->size(); j++) {
          for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
            double element = alpha[j];

            for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {
              element *= ((l2dot(i, j, d_inner) * (d_outer != d_inner)) + (gradient(i, j, d_inner) * (d_outer == d_inner)));
            }

            result[i] += (this->lambda_->get(d_outer) * element);
          }
        }
      }

#endif
    }


    void OperationLaplaceVectorizedLinearLU::generateA(sg::base::DataVector& alpha, double * A) {

      std::cout << " generateA Start" << std::endl;
      for (size_t d = 0; d < this->storage->dim(); d++) {
        sg::base::BoundingBox* boundingBox = this->storage->getBoundingBox();
        lcl_q[d] = boundingBox->getIntervalWidth(d);
        lcl_q_inv[d] = 1.0 / boundingBox->getIntervalWidth(d);
      }



#if 1
      this->OCLPDEKernelsHandle.
	RunOCLKernelGenAInner(alpha, A, lcl_q, lcl_q_inv,
			      this->level_->getPointer(),
			      this->index_->getPointer(),
			      this->level_int_->getPointer(),
			      this->lambda_->getPointer(),
			      storage->size(),
			      storage->dim(),
			      storage);

#else
      //       //#pragma omp parallel for

      for (size_t i = 0; i < this->storage->size(); i++) {
        for (size_t j = 0; j < this->storage->size(); j++) {
	  double res = 0.0;
          for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
            double element = 1.0;//alpha[j];
	    // 	    if (element > 0.0) 
	    // 	      std::cout << " element " << element << std::endl;

            for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {
              element *= ((l2dot(i, j, d_inner) * (d_outer != d_inner)) + (gradient(i, j, d_inner) * (d_outer == d_inner)));
            }
            res += (this->lambda_->get(d_outer) * element);
          }
	  A[i * alpha.getSize() + j] = res;
        }
      }
//       //       std::cout << " generateA End" << std::endl;
//       for(unsigned int i = 350; i < 400; i++) {
// 	std::cout << A[i] << " ";
// 	if (i % 6 == 0) {
// 	  std::cout << std::endl;
// 	}
//       }
//       std::cout << std::endl;
//       std::cout << std::endl;
#endif

      //       for(unsigned int i = 351; i < 401; i++) {
      // 	std::cout << A[i] << " ";
      // 	if (i % 6 == 0) {
      // 	  std::cout << std::endl;
      // 	}
      //       }
      //       std::cout << std::endl;

    } // generateA

  }

}
/*
__kernel void multKernel(__global  REAL* ptrLevel,			
			 	__constant  REAL* ptrLevelIndexLevelintcon,	
			 	__global  REAL* ptrIndex, 			
			 	__global  REAL* ptrLevel_int, 			
			 	__global  REAL* ptrAlpha, 			
			 	__global  REAL* ptrParResult, 			
			 	__constant  REAL* ptrLcl_q,			
			 	__constant  REAL* ptrLambda,			
			 	ulong overallMultOffset,   			
                        	ulong ConstantMemoryOffset) 			
{		
*/
