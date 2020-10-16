#ifndef SUBSPACEITERATIONHPP
#define SUBSPACEITERATIONHPP

#include "shape_calculus.hpp"

/*
 * This function computes the eigensolution for the generalized eigenvalue
 * problem \f$ A v = \lambda B v \f$ starting with the initial guess X_0. The
 * eigensolution contains the eigenvectors corresponding to the m largest
 * eigenvalues where m is the number of columns in the initial guess X_0
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param X0 Initial guess X0
 * @param reltol Relative tolerance used for convergence criterion
 * @param abstol Absolute tolerance used for convergence criterion
 * @param delta Error threshold
 *
 * @return pair of pairs. The first pair contains the eigenvalues and the
 * eigenvectors. The second pair contains a boolean indicating low rank
 * approximation and the number of subspace iterations
 */
std::pair<std::pair<Eigen::VectorXd, Eigen::MatrixXd>,
          std::pair<bool, unsigned>>
SubspaceIteration(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                  const Eigen::MatrixXd &X0, double reltol, double abstol,
                  double delta) {
  // Checking if matrices have the right sizes
  assert(A.rows() == A.cols());
  assert(B.rows() == B.cols());
  assert(B.rows() == A.cols());

  // Getting the number of rows
  unsigned n = A.rows();

  // Matrix containing the current iterate
  Eigen::MatrixXd Xk(X0);
  // Vector containing the current eigenvalues
  Eigen::VectorXd lambdak = Eigen::VectorXd::Random(X0.cols());
  unsigned maxit = 5000; // Maximum iterations for finding the solution

  unsigned it = 1;
  double err;
  double relerr;

  bool debugs = false;
  bool set_random = false;
  bool low_rank = false;
  bool cont_loop;

  // In Manuscript, the starting orthonormal set is "X0" in this code
  do {
    cont_loop = false;
    if (debugs) {
      std::cout << "Iteration no. " << it << std::endl;
    }

    // Increasing the iteration number
    ++it;

    // Solving the linear system
    // Xkplus1bar in the code is equivalent to u_i tilde in the manuscript
    Eigen::MatrixXd Xkplus1bar = B.lu().solve(A * Xk);
    /*if (debugs) {
      std::cout << "Xkplus1bar = \n" << Xkplus1bar << std::endl;
    }*/

    // Getting the smaller matrices for the mXm system
    Eigen::MatrixXd Akplus1 = Xkplus1bar.transpose() * A * Xkplus1bar;
    Eigen::MatrixXd Bkplus1 = Xkplus1bar.transpose() * B * Xkplus1bar;

    // Solving the smaller generalized Eigenvalue problem using Eigen
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
    // Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(Akplus1, Bkplus1);

    // Extracting the eigenvectors and eigenvalues
    Eigen::MatrixXd Qkplus1 = ges.eigenvectors().real();
    Eigen::VectorXd lambdakplus1 = ges.eigenvalues().real();
    if (debugs) {
      // std::cout << "Akplus1 \n" << Akplus1 << std::endl;
      // std::cout << "Bkplus1 \n" << Bkplus1 << std::endl;
      // std::cout << "Orthonormality check \n"
      //          << Qkplus1.transpose() * Bkplus1 * Qkplus1 << std::endl;
      std::cout << "lambdakplus1 = \n" << lambdakplus1 << std::endl;
    }

    Eigen::MatrixXd Xkplus1 = Xkplus1bar * Qkplus1;

    Eigen::VectorXd temp = (lambdak - lambdakplus1).cwiseAbs().array() /
                           lambdak.cwiseAbs().array();

    // Calculating the errors based on new estimates
    relerr = temp.maxCoeff();
    err = (lambdak - lambdakplus1).cwiseAbs().maxCoeff();

    /*if (debugs) {
      std::cout << "Error: " << err << std::endl;
    }*/

    // Updates
    Xk = Xkplus1;
    lambdak = lambdakplus1;

    // Getting the maximum eigenvalue
    double lambdamax = lambdakplus1.cwiseAbs().maxCoeff();

    // Finding all the small eigenvalues
    std::vector<unsigned> Z;
    for (unsigned I = 0; I < X0.cols(); ++I) {
      if (abs(lambdakplus1(I) / lambdamax) < 0.01 * delta) {
        Z.push_back(I);
      }
    }

    // If obtained eigenvalue is small, trying random reshuffling to confirm
    if (Z.size() > 0) {
      if (!set_random) { // Random reshuffling only once
        std::cout << "Random reshuffling once! " << std::endl;
        set_random = true;
        cont_loop = true;

        // Replacing the eigenvectors corresponding to small eigenvalues
        for (unsigned J = 0; J < Z.size(); ++J) {
          // Index to be replaced
          unsigned replace_index = Z[J];
          // Extracting the remaining vectors into basis
          Eigen::MatrixXd basis(X0.rows(), X0.cols() - 1);
          for (unsigned K = 0; K < X0.cols(); ++K) {
            if (K < replace_index)
              basis.col(K) =
                  Xk.col(K) / sqrt(innerPdt(Xk.col(K), Xk.col(K), B));
            if (K > replace_index)
              basis.col(K - 1) =
                  Xk.col(K) / sqrt(innerPdt(Xk.col(K), Xk.col(K), B));
          }

          // Replacing with a random vector orthogonal to the basis
          Eigen::VectorXd randnewvec = Eigen::VectorXd::Random(X0.rows());
          // Making newvec orthogonal to the basis
          Eigen::VectorXd newvec =
              randnewvec - basis * basis.transpose() * B * randnewvec;
          // Normalizing the new vector
          double normnewvec = sqrt(innerPdt(newvec, newvec, B));
          newvec /= normnewvec;
          Xk.col(replace_index) = newvec;
        }

      } else {
        // If small eigenvalue found again, a low rank solution is found
        low_rank = true;
      }
    }

    /*atc = lambdakplus1.cwiseAbs().minCoeff() <
          0.01 * delta * lambdakplus1.cwiseAbs().maxCoeff();
    if (atc && atc_control_inner) {
      std::cout << "Using atc to break from inner loop" << std::endl;
      break;
    }*/

    if (it == maxit) {
      std::cout << "Maximum allowed iterations reached!" << std::endl;
      break;
    }

  } while (cont_loop || (relerr > reltol && err > abstol && !low_rank));

  if (low_rank) {
    std::cout << "Used low rank criterion to break out " << std::endl;
  }

  std::cout << "Subspace iteration finished with " << it << " iterations"
            << std::endl;

  // Sorting the eigenvalues and eigenvectors before returning them
  Eigen::VectorXd abseigs = lambdak.cwiseAbs();
  // Storing the eigenvalues as std vector for sorting
  std::vector<double> temp(abseigs.data(), abseigs.data() + X0.cols());
  // Sorting while storing the indices
  auto idx = sort_indexes(temp);

  // Containers for sorted eigenvalues and eigenvectors
  Eigen::VectorXd sorted_eigs(X0.cols());
  Eigen::MatrixXd sorted_eigvecs(n, X0.cols());
  // Filling the containers with eigenvalues and normalized eigenvectors
  for (unsigned i = 0; i < X0.cols(); ++i) {
    sorted_eigs(i) = abseigs(idx[i]);
    // Normalizing the eigenvectors
    sorted_eigvecs.col(i) =
        Xk.col(idx[i]) / sqrt(innerPdt(Xk.col(idx[i]), Xk.col(idx[i]), B));
  }

  // Preparing the output
  auto firstpart = std::make_pair(sorted_eigs, sorted_eigvecs);
  // auto secondpart = std::make_pair(atc,it);
  auto secondpart = std::make_pair(low_rank, it);
  return std::make_pair(firstpart, secondpart);
}

#endif
