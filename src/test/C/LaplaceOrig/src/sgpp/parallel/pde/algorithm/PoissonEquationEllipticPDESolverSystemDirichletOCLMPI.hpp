/* ****************************************************************************
* Copyright (C) 2011 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
**************************************************************************** */
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETOCLMPI_HPP
#define POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETOCLMPI_HPP

#include "pde/operation/OperationEllipticPDESolverSystemDirichlet.hpp"

namespace sg {
  namespace parallel {

    /**
     * This class uses OperationEllipticPDESolverSystemDirichlet
     * to define a solver system for the Poission Equation.
     *
     * For the mult-routine only the Laplace-Operator is required
     */
    class PoissonEquationEllipticPDESolverSystemDirichletParallelMPI : public sg::pde::OperationEllipticPDESolverSystemDirichlet {
    protected:
      sg::base::OperationMatrix* Laplace_Inner;
      sg::base::OperationMatrix* Laplace_Complete;

      void applyLOperatorComplete(sg::base::DataVector& alpha, sg::base::DataVector& result);

      void applyLOperatorInner(sg::base::DataVector& alpha, sg::base::DataVector& result);

    public:
      /**
       * Constructor
       *
       * @param SparseGrid reference to a sparse grid on which the Poisson Equation should be solved
       * @param rhs the right hand side for solving the elliptic PDE
       */
      PoissonEquationEllipticPDESolverSystemDirichletParallelMPI(sg::base::Grid& SparseGrid, sg::base::DataVector& rhs);

      /**
       * Destructor
       */
      virtual ~PoissonEquationEllipticPDESolverSystemDirichletParallelMPI();
    };

  }
}

#endif /* POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETParallelMPI_HPP */