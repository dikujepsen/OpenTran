/******************************************************************************
* Copyright (C) 2011 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include <mpi.h>
#include "parallel/pde/algorithm/PoissonEquationEllipticPDESolverSystemDirichletOCLMPI.hpp"
#include "base/exception/algorithm_exception.hpp"
#include "pde/operation/PdeOpFactory.hpp"
#include "parallel/operation/ParallelOpFactory.hpp"

using namespace sg::op_factory;

namespace sg {
  namespace parallel {
    PoissonEquationEllipticPDESolverSystemDirichletParallelMPI::PoissonEquationEllipticPDESolverSystemDirichletParallelMPI(sg::base::Grid& SparseGrid, sg::base::DataVector& rhs) : sg::pde::OperationEllipticPDESolverSystemDirichlet(SparseGrid, rhs) {

      this->Laplace_Complete = createOperationLaplaceVectorized(*this->BoundGrid, parallel::OpenCL);
      this->Laplace_Inner = createOperationLaplaceVectorized(*this->InnerGrid,parallel::OpenCL);
    }

    PoissonEquationEllipticPDESolverSystemDirichletParallelMPI::~PoissonEquationEllipticPDESolverSystemDirichletParallelMPI() {
      delete this->Laplace_Complete;
      delete this->Laplace_Inner;
    }

    void PoissonEquationEllipticPDESolverSystemDirichletParallelMPI::applyLOperatorInner(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      Laplace_Inner->mult(alpha, result);

    }

    void PoissonEquationEllipticPDESolverSystemDirichletParallelMPI::applyLOperatorComplete(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      Laplace_Complete->mult(alpha, result);
    }

  }
}
