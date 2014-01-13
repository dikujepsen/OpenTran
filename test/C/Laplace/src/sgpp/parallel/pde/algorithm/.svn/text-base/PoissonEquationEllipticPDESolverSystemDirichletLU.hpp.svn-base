/* ****************************************************************************
* Copyright (C) 2011 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
**************************************************************************** */
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETLU_HPP
#define POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETLU_HPP

#include "pde/operation/OperationEllipticPDESolverSystemDirichlet.hpp"
#include "parallel/pde/basis/linear/noboundary/operation/OperationLaplaceVectorizedLinearLU.hpp"

namespace sg {
  namespace pde {

    /**
     * This class uses OperationEllipticPDESolverSystemDirichlet
     * to define a solver system for the Poission Equation.
     *
     * For the mult-routine only the Laplace-Operator is required
     */
    class PoissonEquationEllipticPDESolverSystemDirichletLU : public OperationEllipticPDESolverSystemDirichlet {
      protected:
        sg::base::OperationMatrix* Laplace_Inner;
        sg::parallel::OperationLaplaceVectorizedLinearLU* Laplace_InnerLU;
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
        PoissonEquationEllipticPDESolverSystemDirichletLU(sg::base::Grid& SparseGrid, sg::base::DataVector& rhs);

      void generateA(sg::base::DataVector& alpha, double * A);
        /**
         * Destructor
         */
        virtual ~PoissonEquationEllipticPDESolverSystemDirichletLU();
    };

  }
}

#endif /* POISSONEQUATIONELLIPTICPDESOLVERSYSTEMDIRICHLETLU_HPP */
