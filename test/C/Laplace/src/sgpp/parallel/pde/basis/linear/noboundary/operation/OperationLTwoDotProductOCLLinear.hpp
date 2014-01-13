/* ****************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
**************************************************************************** */
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef OPERATIONLTWODOTPRODUCTOCLLINEAR_HPP
#define OPERATIONLTWODOTPRODUCTOCLLINEAR_HPP

#include "base/operation/OperationMatrix.hpp"
#include "base/datatypes/DataMatrix.hpp"
#include "base/grid/Grid.hpp"
#include "parallel/pde/basis/common/OCLPDEKernels.hpp" 
#include "parallel/datadriven/basis/common/VectorizedOCLKernels.hpp"

namespace sg {
  namespace parallel {

    /**
     * Implements the standard L 2 scalar product on linear grids (no boundaries)
     *
     * @version $HEAD$
     */
    class OperationLTwoDotProductOCLLinear: public sg::base::OperationMatrix {

    private:
      sg::base::GridStorage* storage;
      sg::base::DataMatrix* level_;
      sg::base::DataMatrix* level_int_;
      sg::base::DataMatrix* index_;
      double* lcl_q;
      OCLPDEKernels OCLPDEKernelsHandle ;

      double l2dot(size_t i, size_t j, size_t dim);
    public:
      /**
       * Constructor
       *
       * @param storage the grid's sg::base::GridStorage object
       */
      OperationLTwoDotProductOCLLinear(sg::base::GridStorage* storage);

      /**
       * Destructor
       */
      virtual ~OperationLTwoDotProductOCLLinear();

    protected:
      virtual void mult(sg::base::DataVector& alpha, sg::base::DataVector& result);


    };

  }
}

#endif /* OPERATIONLTWODOTPRODUCTLINEAR_HPP */
