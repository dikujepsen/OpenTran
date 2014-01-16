/******************************************************************************
* Copyright (C) 2013 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef OPERATIONLAPLACEVECTORIZEDLINEARLU_HPP
#define OPERATIONLAPLACEVECTORIZEDLINEARLU_HPP

#include "base/operation/OperationMatrix.hpp"
#include "base/datatypes/DataMatrix.hpp"
#include "base/grid/Grid.hpp"
#include "parallel/pde/basis/common/OCLPDEKernels.hpp" 


namespace sg {
  namespace parallel {

    /**
     * Implementation for linear functions of Laplace Operation, linear grids without boundaries
     *
     * @version $HEAD$
     */
    class OperationLaplaceVectorizedLinearLU: public sg::base::OperationMatrix {
    private:
      sg::base::GridStorage* storage;
      sg::base::DataMatrix* level_;
      sg::base::DataMatrix* level_int_;
      sg::base::DataMatrix* index_;
      double* lcl_q;
      double* lcl_q_inv;
      sg::base::DataVector* lambda_;
      OCLPDEKernels OCLPDEKernelsHandle ;

      double gradient(size_t i, size_t j, size_t dim);

      double l2dot(size_t i, size_t j, size_t dim);

    public:
      /**
       * Construtor of OperationLaplaceLinear
       *
       * @param storage Pointer to the grid's gridstorage obejct
       */
      OperationLaplaceVectorizedLinearLU(sg::base::GridStorage* storage);

      /**
       * Construtor of OperationLaplaceLinear
       *
       * @param storage Pointer to the grid's gridstorage obejct
       * @param lambda Vector which contains pre-factors for every dimension of the operator
       */
      OperationLaplaceVectorizedLinearLU(sg::base::GridStorage* storage, sg::base::DataVector& lambda);

      void generateA(sg::base::DataVector& alpha, double * A);
      /**
       * Destructor
       */
      virtual ~OperationLaplaceVectorizedLinearLU();

      virtual void mult(sg::base::DataVector& alpha, sg::base::DataVector& result);
    };

  }

}

#endif /* OPERATIONLAPLACEVECTORIZEDLINEARLU_HPP */
