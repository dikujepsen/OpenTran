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
  namespace pde {

    PoissonEquationEllipticPDESolverSystemDirichletOCLMPI::PoissonEquationEllipticPDESolverSystemDirichletOCLMPI(sg::base::Grid& SparseGrid, sg::base::DataVector& rhs) : OperationEllipticPDESolverSystemDirichlet(SparseGrid, rhs) {
      
      // this->Laplace_Complete = createOperationLaplaceVectorized(*this->BoundGrid, parallel::OpenCL);
      // this->Laplace_Inner = createOperationLaplaceVectorized(*this->InnerGrid,parallel::OpenCL);
      // #else
      this->Laplace_Complete = createOperationLaplaceVectorized(*this->BoundGrid,parallel::X86SIMD);
      this->Laplace_Inner = (sg::parallel::OperationLaplaceVectorizedLinear*)createOperationLaplaceVectorized(*this->InnerGrid,parallel::X86SIMD);
    }

    PoissonEquationEllipticPDESolverSystemDirichletOCLMPI::~PoissonEquationEllipticPDESolverSystemDirichletOCLMPI() {
      delete this->Laplace_Complete;
      delete this->Laplace_Inner;
    }

    void PoissonEquationEllipticPDESolverSystemDirichletOCLMPI::applyLOperatorInner(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      // MPIHERE
      size_t result_size = result.getSize();
      int myrank, nproz;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      MPI_Comm_size(MPI_COMM_WORLD, &nproz);
      // Laplace_Inner->mult(alpha, result, 0, result_size);
      //result_size = 350;
      int div = result_size / nproz;
      
      
      // size_t mod = result_size % 2;
      size_t end = (myrank+1) == nproz ? result_size : (myrank+1) * div;
      Laplace_Inner->mult(alpha, result, myrank*div, end );
      
      //Laplace_Inner->mult(alpha, result, div, result_size);
      double * ptrResult = result.getPointer();
      int* sendcnts = new int[nproz];
      int* sdispls = new int[nproz];
      int* recvcnts = new int[nproz];
      int* rdispls = new int[nproz];
      int leftover = result_size;
      int displ = 0;
      for (int i = 0; i < nproz-1; i++) {
	recvcnts[i] = std::min(div, leftover);
	rdispls[i]  = displ;
	displ += recvcnts[i];
	leftover -= recvcnts[i];
      }
      recvcnts[nproz-1] = leftover;
      rdispls[nproz-1] = displ;

      for (int i = nproz-1; i >= 0; i--) {
	sendcnts[i] = recvcnts[myrank];
	sdispls[i] = rdispls[myrank];
      }
      
      // double * ptrResult2 = (double*)malloc(sizeof(double)*result_size);

      // for (size_t i = 0; i < result_size; i++) {
      // 	ptrResult2[i] = ptrResult[i];
      // }
      // MPI_Status status;
      // MPI_Request request,request2;
      // MPI_Isend(ptrResult2 + sdispls[myrank], sendcnts[myrank], MPI_DOUBLE, (myrank + 1) % nproz, 678, MPI_COMM_WORLD, &request);
      // MPI_Irecv(ptrResult + rdispls[myrank], recvcnts[myrank], MPI_DOUBLE, (myrank + 1) % nproz, 678, MPI_COMM_WORLD, &request2);
      // MPI_Wait(&request, &status);
      // MPI_Wait(&request2, &status);
      sg::base::DataVector tmp(result.getSize());
      MPI_Alltoallv(ptrResult,sendcnts, sdispls, MPI_DOUBLE,
		    tmp.getPointer(),recvcnts, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
      result.copyFrom(tmp);
      delete[] recvcnts;
      delete[] rdispls;
      delete[] sendcnts;
      delete[] sdispls;
    }

    void PoissonEquationEllipticPDESolverSystemDirichletOCLMPI::applyLOperatorComplete(sg::base::DataVector& alpha, sg::base::DataVector& result) {
      Laplace_Complete->mult(alpha, result);
    }

  }
}
