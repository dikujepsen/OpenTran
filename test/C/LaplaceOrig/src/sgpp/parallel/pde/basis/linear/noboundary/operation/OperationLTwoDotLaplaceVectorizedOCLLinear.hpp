/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef OPERATIONLTWODOTLAPLACEVECTORIZEDOCLLINEAR_HPP
#define OPERATIONLTWODOTLAPLACEVECTORIZEDOCLLINEAR_HPP

#include "base/operation/OperationMatrix.hpp"
#include "base/datatypes/DataMatrix.hpp"
#include "base/grid/Grid.hpp"
#include "parallel/pde/basis/common/OCLPDEKernels.hpp" 

namespace sg {
  namespace parallel {

    /**
     * Implementation for linear functions of LTwoDotLaplace Operation, linear grids without boundaries
     *
     * @version $HEAD$
     */
    class OperationLTwoDotLaplaceVectorizedOCLLinear: public sg::base::OperationMatrix {
    private:
      sg::base::GridStorage* storage;
      sg::base::DataMatrix* level_;
      sg::base::DataMatrix* level_int_;
      sg::base::DataMatrix* index_;
      double* lcl_q;
      double* lcl_q_inv;
      sg::base::DataVector* lambda;
      REAL TimestepCoeff;

      OCLPDEKernels OCLPDEKernelsHandle ;
      size_t padding_size;
      size_t sizepad;
      double* subresult;

      double gradient(
		      double i_level_grad,
		      double i_index_grad,
		      double j_level_grad,
		      double j_index_grad
		      );

      double l2dot(
	   double level_i,double index_i,double level_int_i,
	   double level_j,double index_j,double level_int_j, size_t d);
    public:
      /**
       * Construtor of OperationLTwoDotLaplaceLinear
       *
       * @param storage Pointer to the grid's gridstorage obejct
       * @param lambda Vector which contains pre-factors for every dimension of the operator
       */
      OperationLTwoDotLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage, sg::base::DataVector& lambda);

      /**
       * Construtor of OperationLTwoDotLaplaceLinear
       *
       * @param storage Pointer to the grid's gridstorage obejct
       */
      OperationLTwoDotLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage);

      /**
       * Destructor
       */
      virtual ~OperationLTwoDotLaplaceVectorizedOCLLinear();

      virtual void mult(sg::base::DataVector& alpha, 
			sg::base::DataVector& result);

      /**
       * Sets the timestep coefficient
       *
       * @param newTimestepCoeff The new timestep coefficient for the chosen 
       * numerical approximation scheme. 
       */
      void SetTimestepCoeff(REAL newTimestepCoeff);

      /**
       * Gets the timestep coefficient
       *
       * @return newTimestepCoeff The new timestep coefficient for the chosen 
       * numerical approximation scheme. 
       */
      REAL GetTimestepCoeff();

    };

  }

}

#endif /* OPERATIONLTWODOTLAPLACEVECTORIZEDOCLLINEAR_HPP */