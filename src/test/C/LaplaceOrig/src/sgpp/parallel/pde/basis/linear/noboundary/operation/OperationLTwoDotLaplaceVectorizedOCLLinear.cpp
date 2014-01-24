/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "base/grid/type/LinearGrid.hpp"
#include "base/grid/generation/GridGenerator.hpp"

#include "parallel/pde/basis/linear/noboundary/operation/OperationLTwoDotLaplaceVectorizedOCLLinear.hpp"

#include <cmath>
#define GPU1 1

namespace sg {
  namespace parallel {

    OperationLTwoDotLaplaceVectorizedOCLLinear::OperationLTwoDotLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage, sg::base::DataVector& lambda) : storage(storage){


      this->TimestepCoeff = 0.0;
      this->lambda = new sg::base::DataVector(lambda);
      this->OCLPDEKernelsHandle = OCLPDEKernels();
      padding_size = 16;
      //       size_t size = storage->size();
      //       size_t pad = padding_size - (size % padding_size);
      sizepad = storage->size();
      //       std::cout << " sizepad " << sizepad << std::endl;
      this->level_ = new sg::base::DataMatrix(sizepad, storage->dim());
      //       this->subresult = new double[sizepad * sizepad];
      this->level_int_ = new sg::base::DataMatrix(sizepad, storage->dim());
      this->index_ = new sg::base::DataMatrix(sizepad, storage->dim());
      lcl_q = new double[this->storage->dim()];
      lcl_q_inv = new double[this->storage->dim()];

      storage->getLevelIndexArraysForEval(*(this->level_), *(this->index_));
      storage->getLevelForIntegral(*(this->level_int_));



    }

    OperationLTwoDotLaplaceVectorizedOCLLinear::OperationLTwoDotLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage) : storage(storage) {

      
      this->TimestepCoeff = 0.0;
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


    OperationLTwoDotLaplaceVectorizedOCLLinear::~OperationLTwoDotLaplaceVectorizedOCLLinear() {
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

    double OperationLTwoDotLaplaceVectorizedOCLLinear::gradient(
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

    double OperationLTwoDotLaplaceVectorizedOCLLinear::l2dot(
							     double level_i,double index_i,double level_int_i,
							     double level_j,double index_j,double level_int_j,size_t d) {
      double lid = level_i;
      double ljd = level_j;
      double iid = index_i;
      double ijd = index_j;
      double in_lid = level_int_i;
      double in_ljd = level_int_j;

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

      return (res_one * (lid == ljd) + res_two * (lid != ljd)) * lcl_q[d];
    }

    void OperationLTwoDotLaplaceVectorizedOCLLinear::mult(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      result.setAll(0.0);

      // fill q array
      for (size_t d = 0; d < this->storage->dim(); d++) {
	sg::base::BoundingBox* boundingBox = this->storage->getBoundingBox();
	lcl_q[d] = boundingBox->getIntervalWidth(d);
	lcl_q_inv[d] = 1.0 / boundingBox->getIntervalWidth(d);
      }
      //       std::cout << " TimestepCoeff " << TimestepCoeff << std::endl;

#if GPU1
      
      this->OCLPDEKernelsHandle.
	RunOCLKernelLTwoDotLaplaceInner(alpha, result, 
						 this->lcl_q, this->lcl_q_inv,
						 this->level_->getPointer(),
						 this->index_->getPointer(),
						 this->level_int_->getPointer(),
						 this->lambda->getPointer(),
						 this->storage->size(),
						 this->storage->dim(),
						 this->storage,
						 this->TimestepCoeff);
#else //1:            //        0.046871
      //2: 0.0412421  // Rigtig 0.0419265 
      //3: 0.0448693  //        0.0448693
      //4: 0.0370874  //        0.0370874
      //5: 0.0353998  23.2674(loop)/21.0119(unroll)/18.73(global memory)//        0.0353998 // 32.9722
#if 1
#pragma omp parallel
      {
	double* gradient_temp = new double[this->storage->dim()];
	double* dot_temp = new double[this->storage->dim()];
#pragma omp for

	for (size_t ii = 0; ii < this->storage->size(); ii++) {
	  double LTwoDotLaplaceRes = 0.0;
	  for (size_t jj = 0; jj < this->storage->size(); jj++) {

	    for (size_t d = 0; d < this->storage->dim(); d++) {

	      double level_i = level_->get(ii, d);
	      double index_i = index_->get(ii, d);
	      double level_int_i = level_int_->get(ii, d);
	      double level_j = level_->get(jj, d);
	      double index_j = index_->get(jj, d);
	      double level_int_j = level_int_->get(jj, d);
	      gradient_temp[d] = gradient(level_i,index_i,
					  level_j,index_j) * lcl_q_inv[d];
	      dot_temp[d] = l2dot(level_i,index_i,level_int_i,
				  level_j,index_j,level_int_j,
				  d);
	      // 	      dot_temp2[d] = l2dot(level_j,index_j,level_int_j,
	      // 				   level_i,index_i,level_int_i,
	      // 				   d);
	      // 	      if ( dot_temp[d] != dot_temp2[d] ) {
	      // 		std::cout << dot_temp[d] << " != " << dot_temp2[d] << std::endl;
	      // 	    }
	    }

	    double element = 0.0;
	    for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
	      element = alpha[jj];

	      for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {

		element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (d_outer == d_inner));
	      }
	      LTwoDotLaplaceRes += this->TimestepCoeff * lambda->get(d_outer) * element * (gradient_temp[d_outer]);
	    }
	    LTwoDotLaplaceRes += element * dot_temp[this->storage->dim()-1];
            
	  }
	  result[ii] += LTwoDotLaplaceRes;
	}
	delete [] gradient_temp;
	delete [] dot_temp;

      }
#else
#pragma omp parallel
      {
	double* gradient_temp = new double[this->storage->dim()];
	double* dot_temp = new double[this->storage->dim()];
	// 	std::cout << " NEW " << std::endl;
#pragma omp for
	for (size_t bb = 0; bb < sizepad; bb+=padding_size) {

	  // 	    std::cout << " iisize: " << sizepad-bb <<std::endl;
	  for (size_t ii = bb; ii < sizepad; ii++) {
	    double LTwoDotLaplaceRes = 0.0;
	    for (size_t jj = bb; jj <(bb+padding_size) ; jj++) {
	      // 		std::cout << " ["<<ii<<","<<jj<<"] ";
	      double PartialRes = 0.0;

	      for (size_t d = 0; d < this->storage->dim(); d++) {

		double level_i = level_->get(ii, d);
		double index_i = index_->get(ii, d);
		double level_int_i = level_int_->get(ii, d);
		double level_j = level_->get(jj, d);
		double index_j = index_->get(jj, d);
		double level_int_j = level_int_->get(jj, d);
		gradient_temp[d] = gradient(level_i,index_i,
					    level_j,index_j) * lcl_q_inv[d];
		dot_temp[d] = l2dot(level_i,index_i,level_int_i,
				    level_j,index_j,level_int_j,
				    d);
	      }

	      double element = 0.0;
	      for (size_t d_outer = 0; d_outer < this->storage->dim(); d_outer++) {
		element = 1.0;

		for (size_t d_inner = 0; d_inner < this->storage->dim(); d_inner++) {

		  element *= ((dot_temp[d_inner] * (d_outer != d_inner)) + (d_outer == d_inner));
		}
		PartialRes += this->TimestepCoeff * lambda->get(d_outer) * element * (gradient_temp[d_outer]);
		// LTwoDotLaplaceRes += this->TimestepCoeff * lambda->get(d_outer) * element * (gradient_temp[d_outer]) * alpha[jj];
		// if (ii!=jj) {
		// 		  subresult[jj] += this->TimestepCoeff * lambda->get(d_outer) * element * (gradient_temp[d_outer]) * alpha[ii];
		// 		}
	      }
	      PartialRes += element * dot_temp[this->storage->dim()-1];
	      // LTwoDotLaplaceRes += element * dot_temp[this->storage->dim()-1] * alpha[jj];
	      if (jj < this->storage->size()){
		LTwoDotLaplaceRes += PartialRes * alpha[jj] * (jj <= ii);
	      }
	      if (ii < this->storage->size()){
		subresult[jj] += PartialRes * alpha[ii] * (jj < ii);
	      }
	      //  if (ii!=jj) {
	      // 		subresult[jj] += element * dot_temp[this->storage->dim()-1] * alpha[ii];

	      // 	      }
	    } // jj
	    if (ii < this->storage->size()){
	      result[ii] += LTwoDotLaplaceRes;// + subresult[ii];
	    }
	  } // ii
	  for (size_t jj = bb; jj <std::min<size_t>(bb+padding_size, this->storage->size()) ; jj++) {
	    result[jj] += subresult[jj];
	  }
	} // bb
	delete [] gradient_temp;
	delete [] dot_temp;

      } // Parallel

#endif

	
#endif
      //result.copyFrom(temp2);



    }

    void OperationLTwoDotLaplaceVectorizedOCLLinear::SetTimestepCoeff(REAL newTimestepCoeff) {
      this->TimestepCoeff = newTimestepCoeff;
    }
    REAL OperationLTwoDotLaplaceVectorizedOCLLinear::GetTimestepCoeff() {
      return this->TimestepCoeff;
    }


  }

}
