/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "parallel/pde/application/PoissonEquationSolverLU.hpp"
#include "parallel/pde/algorithm/PoissonEquationEllipticPDESolverSystemDirichletLU.hpp"
#include "pde/algorithm/PoissonEquationEllipticPDESolverSystemDirichlet.hpp"
#include "solver/sle/ConjugateGradients.hpp"
#include "base/grid/Grid.hpp"
#include "base/exception/application_exception.hpp"
#include "base/tools/SGppStopwatch.hpp"
#include "base/operation/BaseOpFactory.hpp"
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

#include "testings.h"

using namespace sg::solver;
using namespace sg::base;

void init_matrix( int m, int n, double *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
}


namespace sg {
  namespace pde {

    PoissonEquationSolverLU::PoissonEquationSolverLU() : EllipticPDESolver() {
      std::cout << " Solve Poisson Equation Using LU Decomposiion (PoissonEquationSolverLU)" << std::endl;

      this->bGridConstructed = false;
      this->myScreen = NULL;
    }

    PoissonEquationSolverLU::~PoissonEquationSolverLU() {
      if (this->myScreen != NULL) {
        delete this->myScreen;
      }
    }

    void PoissonEquationSolverLU::constructGrid(BoundingBox& BoundingBox, int level) {
      this->dim = BoundingBox.getDimensions();
      this->levels = level;

      this->myGrid = new LinearTrapezoidBoundaryGrid(BoundingBox);

      GridGenerator* myGenerator = this->myGrid->createGridGenerator();
      myGenerator->regular(this->levels);
      delete myGenerator;

      this->myBoundingBox = this->myGrid->getBoundingBox();
      this->myGridStorage = this->myGrid->getStorage();

      this->bGridConstructed = true;
    }

    void PoissonEquationSolverLU::solvePDE(DataVector& alpha, DataVector& rhs, size_t maxCGIterations, double epsilonCG, bool verbose) {
      double dTimeAlpha = 0.0;
      double dTimeRHS = 0.0;
      double dTimeSolver = 0.0;

      SGppStopwatch* myStopwatch = new SGppStopwatch();
      ConjugateGradients* myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
      PoissonEquationEllipticPDESolverSystemDirichletLU* mySystem = new PoissonEquationEllipticPDESolverSystemDirichletLU(*(this->myGrid), rhs);

      std::cout << "Gridpoints (complete grid): " << mySystem->getNumGridPointsComplete() << std::endl;
      std::cout << "Gridpoints (inner grid): " << mySystem->getNumGridPointsInner() << std::endl << std::endl << std::endl;

      myStopwatch->start();
      DataVector* alpha_solve = mySystem->getGridCoefficientsForCG();
      dTimeAlpha = myStopwatch->stop();
      std::cout << "coefficients has been initialized for solving!" << std::endl;

      //       double * ptrAlpha = alpha_solve->getPointer();
      //       for (int i = 0; i < 50; i ++) {
      // 	std::cout << ptrAlpha[i] << " ";
      // 	if (i % 6 == 0) {
      // 	  std::cout << std::endl;
      // 	}
      //       }

      myStopwatch->start();
      DataVector* rhs_solve = mySystem->generateRHS();
      dTimeRHS = myStopwatch->stop();
      std::cout << "right hand side has been initialized for solving!" << std::endl << std::endl << std::endl;

      myStopwatch->start();

      double *h_A, *h_R;
      double *d_A;
      DataVector* datavector_x = new DataVector(*alpha_solve);
      double *x = datavector_x->getPointer();

      double * ptrRHS = rhs_solve->getPointer();
      magma_int_t M = 0, N = 0, lda, ldda, n2, info, min_mn, *ipiv;
      const double c_one     = MAGMA_D_ONE;
      const double c_neg_one = MAGMA_D_NEG_ONE;
      const magma_int_t ione = 1;
      N = rhs_solve->getSize(); // Size of RHS
      M = alpha_solve->getSize();  // Other side of the matrix
      //       std:cout << " N " << N << std::endl;
      //       std::cout << " M " << M << std::endl;
      lda = M;
      n2     = lda * N;
      min_mn = min(M, N);
      info = 2;
      ldda   = ((M+31)/32)*32;
      TESTING_HOSTALLOC(    h_A,  double, n2 );

      //       TESTING_MALLOC( x, double, N );
      //init_matrix( M, N, h_A, lda );
      //       for (int i = 0; i < 50; i ++) {
      // 	std::cout << h_A[i] << " ";
      // 	if (i % 6 == 0) {
      // 	  std::cout << std::endl;
      // 	}
      //       }
#define LU 1
#if LU
      real_Double_t   gen_time, gpu_time, matmul_time;
      gen_time = magma_wtime();
      mySystem->generateA(*alpha_solve,h_A);
      gen_time = magma_wtime() - gen_time;

      //LU Decomposition
      TESTING_CUDA_INIT();
      TESTING_MALLOC(ipiv, magma_int_t, min_mn);
      TESTING_HOSTALLOC( h_R,  double, n2     );
      TESTING_DEVALLOC(  d_A,  double, ldda*N );

      gpu_time = magma_wtime();
#if 1
      lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
      magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
      magma_dgetrf_gpu( M, N, d_A, ldda, ipiv, &info);
      magma_dgetmatrix( M, N, d_A, ldda, h_A, lda );
      TESTING_HOSTFREE( h_R );
      TESTING_DEVFREE( d_A );

#else
      //       lapackf77_dgetrf(&M, &N, h_A, &lda, ipiv, &info);
      magma_dgetrf( M, N, h_A, lda, ipiv, &info);
#endif
      if (info != 0)
	printf("magma_dgetrf returned error %d: %s.\n",
	       (int) info, "info");

      gpu_time = magma_wtime() - gpu_time;

      blasf77_dcopy( &N, ptrRHS, &ione, x, &ione );


      // Solve Ax = b
      matmul_time = magma_wtime();
      lapackf77_dgetrs( "Notrans", &N, &ione, h_A, &lda, ipiv, x, &N, &info );
      matmul_time = magma_wtime() - matmul_time;
      if (info != 0)
	printf("magma_dgetrs returned error %d: %s.\n",
	       (int) info, "magma_strerror( info )");

      double* ptrAlpha = datavector_x->getPointer();
      for (int i = 0; i < 50; i ++) {
      	std::cout << ptrAlpha[i] << " ";
      	if (i % 6 == 0) {
      	  std::cout << std::endl;
      	}
      }
      std::cout << std::endl;
      std::cout << std::endl;
      mySystem->getSolutionBoundGrid(alpha, *datavector_x);
      std::cout << " GEN: " << gen_time << std::endl;
      std::cout << " LU DECOMP: " << gpu_time << std::endl;
      std::cout << " MATMUL: " << matmul_time << std::endl;
      TESTING_FREE( ipiv );
      TESTING_HOSTFREE( h_A );
      TESTING_CUDA_FINALIZE();

#endif
#if 0
      // Reset to Original h_A in order to compute residual
      //init_matrix( M, N, h_A, lda );
      mySystem->generateA(*alpha_solve,h_A);
      
      // compute r = Ax - b, saved in b
      blasf77_dgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_neg_one, ptrRHS, &ione );
      
      for (int i = 0; i < 50; i ++) {
	std::cout << ptrRHS[i] << " ";
	if (i % 6 == 0) {
	  std::cout << std::endl;
	}
      }
#endif

      // for (int i = 0; i < 50; i ++) {
      // 	std::cout << x[i] << " ";
      // 	if (i % 6 == 0) {
      // 	  std::cout << std::endl;
      // 	}
      //       }

#if !LU
      myCG->solve(*mySystem, *alpha_solve, *rhs_solve, true, verbose, 0.0);

      double* ptrAlpha = alpha_solve->getPointer();
      for (int i = 0; i < 50; i ++) {
      	std::cout << ptrAlpha[i] << " ";
      	if (i % 6 == 0) {
      	  std::cout << std::endl;
      	}
      }
      std::cout << std::endl;
      std::cout << std::endl;
      
      // Copy result into coefficient vector of the boundary grid
      mySystem->getSolutionBoundGrid(alpha, *alpha_solve);
#endif
      dTimeSolver = myStopwatch->stop();

      //       TESTING_FREE( x );


      //       std::cout << std::endl << std::endl;
      //       std::cout << "Gridpoints (complete grid): " << mySystem->getNumGridPointsComplete() << std::endl;
      //       std::cout << "Gridpoints (inner grid): " << mySystem->getNumGridPointsInner() << std::endl << std::endl << std::endl;

      std::cout << "Timings for solving Poisson Equation" << std::endl;
      std::cout << "------------------------------------" << std::endl;
      std::cout << "Time for creating CG coeffs: " << dTimeAlpha << std::endl;
      std::cout << "Time for creating RHS: " << dTimeRHS << std::endl;
      std::cout << "Time for solve: " << dTimeSolver << std::endl << std::endl;
      std::cout << "Time: " << dTimeAlpha + dTimeRHS + dTimeSolver << std::endl << std::endl << std::endl;

      delete myCG;
      delete mySystem; // alpha_solver and rhs_solve are allocated and freed here!!
      delete myStopwatch;
    }

    void PoissonEquationSolverLU::initGridWithSmoothHeat(DataVector& alpha, double mu, double sigma, double factor) {
      if (this->bGridConstructed) {
        double tmp;
        double* dblFuncValues = new double[this->dim];

        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++) {
          std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
          std::stringstream coordsStream(coords);
          bool isInner = true;

          for (size_t j = 0; j < this->dim; j++) {
            coordsStream >> tmp;

            // determine if a grid point is an inner grid point
            if ((tmp != this->myBoundingBox->getBoundary(j).leftBoundary && tmp != this->myBoundingBox->getBoundary(j).rightBoundary)) {
              // Nothtin to do, test is that qay hence == for floating point values is unsave
            } else {
              isInner = false;
            }

            dblFuncValues[j] = tmp;
          }

          if (isInner == false) {
            tmp = 1.0;

            for (size_t j = 0; j < this->dim; j++) {
              tmp *=  factor * factor * ((1.0 / (sigma * 2.0 * 3.145)) * exp((-0.5) * ((dblFuncValues[j] - mu) / sigma) * ((dblFuncValues[j] - mu) / sigma)));
            }
          } else {
            tmp = 0.0;
          }

          alpha[i] = tmp;
        }

        delete[] dblFuncValues;

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      } else {
        throw new application_exception("HeatEquationSolver::initGridWithSmoothHeat : A grid wasn't constructed before!");
      }
    }

    void PoissonEquationSolverLU::initGridWithSmoothHeatFullDomain(DataVector& alpha, double mu, double sigma, double factor) {
      if (this->bGridConstructed) {
        double tmp;
        double* dblFuncValues = new double[this->dim];

        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++) {
          std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
          std::stringstream coordsStream(coords);

          for (size_t j = 0; j < this->dim; j++) {
            coordsStream >> tmp;

            dblFuncValues[j] = tmp;
          }

          tmp = 1.0;

          for (size_t j = 0; j < this->dim; j++) {
            tmp *=  factor * factor * ((1.0 / (sigma * 2.0 * 3.145)) * exp((-0.5) * ((dblFuncValues[j] - mu) / sigma) * ((dblFuncValues[j] - mu) / sigma)));
          }

          alpha[i] = tmp;
        }

        delete[] dblFuncValues;

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      } else {
        throw new application_exception("HeatEquationSolver::initGridWithSmoothHeatFullDomain : A grid wasn't constructed before!");
      }
    }

    void PoissonEquationSolverLU::initGridWithExpHeat(DataVector& alpha, double factor) {
      if (this->bGridConstructed) {
        double tmp;
        double* dblFuncValues = new double[this->dim];
        double* rightBound = new double[this->dim];

        BoundingBox* tmpBB = this->myGrid->getBoundingBox();

        for (size_t j = 0; j < this->dim; j++) {
          rightBound[j] = (tmpBB->getBoundary(j)).rightBoundary;
        }

        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++) {
          std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
          std::stringstream coordsStream(coords);
          bool isInner = true;
          tmp = 0.0;

          for (size_t j = 0; j < this->dim; j++) {
            coordsStream >> tmp;

            // determine if a grid point is an inner grid point
            if ((tmp != this->myBoundingBox->getBoundary(j).leftBoundary && tmp != this->myBoundingBox->getBoundary(j).rightBoundary)) {
              // Nothtin to do, test is that qay hence == for floating point values is unsave
            } else {
              isInner = false;
            }

            dblFuncValues[j] = tmp;
          }

          if (isInner == false) {
            tmp = 1.0;

            for (size_t j = 0; j < this->dim; j++) {
              tmp *= exp((dblFuncValues[j] - rightBound[j]) * factor);
            }
          } else {
            tmp = 0.0;
          }

          alpha[i] = tmp;
        }

        delete[] dblFuncValues;

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      } else {
        throw new application_exception("PoissonEquationSolver::initGridWithExpHeat : A grid wasn't constructed before!");
      }
    }

    void PoissonEquationSolverLU::initGridWithExpHeatFullDomain(DataVector& alpha, double factor) {
      if (this->bGridConstructed) {
        double tmp;
        double* dblFuncValues = new double[this->dim];
        double* rightBound = new double[this->dim];

        BoundingBox* tmpBB = this->myGrid->getBoundingBox();

        for (size_t j = 0; j < this->dim; j++) {
          rightBound[j] = (tmpBB->getBoundary(j)).rightBoundary;
        }

        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++) {
          std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
          std::stringstream coordsStream(coords);
          tmp = 0.0;

          for (size_t j = 0; j < this->dim; j++) {
            coordsStream >> tmp;

            dblFuncValues[j] = tmp;
          }

          tmp = 1.0;

          for (size_t j = 0; j < this->dim; j++) {
            tmp *= exp((dblFuncValues[j] - rightBound[j]) * factor);
          }

          alpha[i] = tmp;
        }

        delete[] dblFuncValues;

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      } else {
        throw new application_exception("PoissonEquationSolver::initGridWithExpHeat : A grid wasn't constructed before!");
      }
    }

    void PoissonEquationSolverLU::storeInnerMatrix(std::string tFilename) {
      DataVector rhs(this->myGrid->getSize());
      rhs.setAll(0.0);
      PoissonEquationEllipticPDESolverSystemDirichlet* mySystem = new PoissonEquationEllipticPDESolverSystemDirichlet(*(this->myGrid), rhs);
      SGppStopwatch* myStopwatch = new SGppStopwatch();

      std::string mtx = "";

      std::cout << "Generating matrix in MatrixMarket format..." << std::endl;
      myStopwatch->start();
      mySystem->getInnerMatrix(mtx);

      std::ofstream outfile(tFilename.c_str());
      outfile << mtx;
      outfile.close();
      std::cout << "Generating matrix in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

      delete myStopwatch;
      delete mySystem;
    }

    void PoissonEquationSolverLU::storeInnerMatrixDiagonal(std::string tFilename) {
      DataVector rhs(this->myGrid->getSize());
      rhs.setAll(0.0);
      PoissonEquationEllipticPDESolverSystemDirichlet* mySystem = new PoissonEquationEllipticPDESolverSystemDirichlet(*(this->myGrid), rhs);
      SGppStopwatch* myStopwatch = new SGppStopwatch();

      std::string mtx = "";

      std::cout << "Generating diagonal matrix in MatrixMarket format..." << std::endl;
      myStopwatch->start();
      mySystem->getInnerMatrixDiagonal(mtx);

      std::ofstream outfile(tFilename.c_str());
      outfile << mtx;
      outfile.close();
      std::cout << "Generating diagonal matrix in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

      delete myStopwatch;
      delete mySystem;
    }

    void PoissonEquationSolverLU::storeInnerMatrixDiagonalRowSum(std::string tFilename) {
      DataVector rhs(this->myGrid->getSize());
      rhs.setAll(0.0);
      PoissonEquationEllipticPDESolverSystemDirichlet* mySystem = new PoissonEquationEllipticPDESolverSystemDirichlet(*(this->myGrid), rhs);
      SGppStopwatch* myStopwatch = new SGppStopwatch();

      std::string mtx = "";

      std::cout << "Generating row sum diagonal matrix in MatrixMarket format..." << std::endl;
      myStopwatch->start();
      mySystem->getInnerMatrixDiagonalRowSum(mtx);

      std::ofstream outfile(tFilename.c_str());
      outfile << mtx;
      outfile.close();
      std::cout << "Generating row sum diagonal matrix in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

      delete myStopwatch;
      delete mySystem;
    }

    void PoissonEquationSolverLU::storeInnerRHS(DataVector& alpha, std::string tFilename) {
      SGppStopwatch* myStopwatch = new SGppStopwatch();
      PoissonEquationEllipticPDESolverSystemDirichlet* mySystem = new PoissonEquationEllipticPDESolverSystemDirichlet(*(this->myGrid), alpha);

      std::cout << "Exporting inner right-hand-side..." << std::endl;
      myStopwatch->start();
      DataVector* rhs_inner = mySystem->generateRHS();

      size_t nCoefs = rhs_inner->getSize();
      std::ofstream outfile(tFilename.c_str());

      for (size_t i = 0; i < nCoefs; i++) {
        outfile << std::scientific << rhs_inner->get(i) << std::endl;
      }

      outfile.close();
      std::cout << "Exporting inner right-hand-side... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

      delete mySystem; // rhs_inner are allocated and freed here!!
      delete myStopwatch;
    }

    void PoissonEquationSolverLU::storeInnerSolution(DataVector& alpha, size_t maxCGIterations, double epsilonCG, std::string tFilename) {
      ConjugateGradients* myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
      PoissonEquationEllipticPDESolverSystemDirichlet* mySystem = new PoissonEquationEllipticPDESolverSystemDirichlet(*(this->myGrid), alpha);

      std::cout << "Exporting inner solution..." << std::endl;

      DataVector* alpha_solve = mySystem->getGridCoefficientsForCG();
      DataVector* rhs_solve = mySystem->generateRHS();

      myCG->solve(*mySystem, *alpha_solve, *rhs_solve, true, false, 0.0);

      size_t nCoefs = alpha_solve->getSize();
      std::ofstream outfile(tFilename.c_str());

      for (size_t i = 0; i < nCoefs; i++) {
        outfile << std::scientific << alpha_solve->get(i) << std::endl;
      }

      outfile.close();

      std::cout << "Exporting inner solution... DONE!" << std::endl;

      delete myCG;
      delete mySystem; // alpha_solver and rhs_solve are allocated and freed here!!
    }

    void PoissonEquationSolverLU::initScreen() {
      this->myScreen = new ScreenOutput();
      this->myScreen->writeTitle("SGpp - Poisson Equation SolverLU, 1.0.0", "Alexander Heinecke, (C) 2009-2011");
    }

  }
}
