/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef OPERATIONLAPLACEVECTORIZEDOCLLINEAR_HPP
#define OPERATIONLAPLACEVECTORIZEDOCLLINEAR_HPP

#include "base/operation/OperationMatrix.hpp"
#include "base/datatypes/DataMatrix.hpp"
#include "base/grid/Grid.hpp"
#include "parallel/pde/basis/common/OCLPDEKernels.hpp" 
// #include "parallel/datadriven/basis/common/VectorizedOCLKernels.hpp"

namespace sg {
  namespace parallel {

    /**
     * Implementation for linear functions of Laplace Operation, linear grids without boundaries
     *
     * @version $HEAD$
     */
    class OperationLaplaceVectorizedOCLLinear: public sg::base::OperationMatrix {
    private:
      sg::base::GridStorage* storage;
      sg::base::DataMatrix* level_;
      sg::base::DataMatrix* level_int_;
      sg::base::DataMatrix* index_;
      double* lcl_q;
      double* lcl_q_inv;
      sg::base::DataVector* lambda;


      OCLPDEKernels OCLPDEKernelsHandle ;

      double gradient(
		      double i_level_grad,
		      double i_index_grad,
		      double j_level_grad,
		      double j_index_grad
		                );

	double l2dot(size_t i, size_t j, size_t dim);

      public:
      /**
       * Construtor of OperationLaplaceLinear
       *
       * @param storage Pointer to the grid's gridstorage obejct
       * @param lambda Vector which contains pre-factors for every dimension of the operator
       */
	OperationLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage, sg::base::DataVector& lambda);

	/**
	 * Construtor of OperationLaplaceLinear
	 *
	 * @param storage Pointer to the grid's gridstorage obejct
	 */
	OperationLaplaceVectorizedOCLLinear(sg::base::GridStorage* storage);

	/**
	 * Destructor
	 */
	virtual ~OperationLaplaceVectorizedOCLLinear();

	virtual void mult(sg::base::DataVector& alpha, sg::base::DataVector& result);
      };

    }

  }

#endif /* OPERATIONLAPLACEVECTORIZEDOCLLINEAR_HPP */
