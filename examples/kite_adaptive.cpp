#include "functionals.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

std::pair<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, std::pair<bool,unsigned>>
SubspaceIteration(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                  const Eigen::MatrixXd &X0, double reltol, double abstol,
                  double delta) {
  //
  assert(A.rows() == A.cols());
  assert(B.rows() == B.cols());
  assert(B.rows() == A.cols());

  unsigned n = A.rows();

  Eigen::MatrixXd Xk(X0);
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
      //std::cout << "Xk = \n" << Xk << std::endl;
    }
    ++it;
    // Xkplus1bar in the code is equivalent to u_i tilde in the manuscript
    Eigen::MatrixXd Xkplus1bar = B.lu().solve(A * Xk);
    /*if (debugs) {
      std::cout << "Xkplus1bar = \n" << Xkplus1bar << std::endl;
    }*/
    Eigen::MatrixXd Akplus1 = Xkplus1bar.transpose() * A * Xkplus1bar;
    Eigen::MatrixXd Bkplus1 = Xkplus1bar.transpose() * B * Xkplus1bar;

    // Solving the generalized Eigenvalue problem using Eigen
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
    //Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(Akplus1, Bkplus1);

    Eigen::MatrixXd Qkplus1 = ges.eigenvectors().real();
    Eigen::VectorXd lambdakplus1 = ges.eigenvalues().real();
    if (debugs) {
      //std::cout << "Akplus1 \n" << Akplus1 << std::endl;
      //std::cout << "Bkplus1 \n" << Bkplus1 << std::endl;
      //std::cout << "Orthonormality check \n"
      //          << Qkplus1.transpose() * Bkplus1 * Qkplus1 << std::endl;
      std::cout << "lambdakplus1 = \n" << lambdakplus1 << std::endl;
    }

    Eigen::MatrixXd Xkplus1 = Xkplus1bar * Qkplus1;

    Eigen::VectorXd temp = (lambdak - lambdakplus1).cwiseAbs().array() /
                           lambdak.cwiseAbs().array();
    relerr = temp.maxCoeff();
    err = (lambdak - lambdakplus1).cwiseAbs().maxCoeff();

    /*if (debugs) {
      std::cout << "Error: " << err << std::endl;
    }*/

    Xk = Xkplus1;
    lambdak = lambdakplus1;

    double lambdamax = lambdakplus1.cwiseAbs().maxCoeff();

    std::vector<unsigned> Z;
    for (unsigned I = 0 ; I < X0.cols() ; ++I) {
      if (abs(lambdakplus1(I)/lambdamax) < 0.01 * delta) {
        Z.push_back(I);
      }
    }

    if (Z.size() > 0) {
      if (!set_random) { // Random reshuffling once
        std::cout << "Random reshuffling once! " << std::endl;
        set_random = true;
        cont_loop = true;

        // Replacing the corresponding vectors
        for (unsigned J = 0 ; J < Z.size() ; ++J) {
          unsigned replace_index = Z[J];
          Eigen::MatrixXd basis(X0.rows(),X0.cols()-1);
          for (unsigned K = 0 ; K < X0.cols() ; ++K) {
            if (K < replace_index)
              basis.col(K) = Xk.col(K)/sqrt(innerPdt(Xk.col(K),Xk.col(K),B));
            if (K > replace_index)
              basis.col(K-1) = Xk.col(K)/sqrt(innerPdt(Xk.col(K),Xk.col(K),B));
          }
          Eigen::VectorXd randnewvec = Eigen::VectorXd::Random(X0.rows());
          // Making newvec orthogonal to the basis
          Eigen::VectorXd newvec = randnewvec - basis * basis.transpose() * B * randnewvec;
          double normnewvec = sqrt(innerPdt(newvec,newvec,B));
          newvec /= normnewvec;
          Xk.col(replace_index) = newvec;
        }

      }
      else {
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

  } while (cont_loop || (relerr > reltol  && err > abstol && !low_rank) );

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
  Eigen::VectorXd sorted_eigs(X0.cols());
  Eigen::MatrixXd sorted_eigvecs(n, X0.cols());

  for (unsigned i = 0; i < X0.cols(); ++i) {
    sorted_eigs(i) = abseigs(idx[i]);
    sorted_eigvecs.col(i) =
        Xk.col(idx[i]) / sqrt(innerPdt(Xk.col(idx[i]), Xk.col(idx[i]), B));
  }

  auto firstpart = std::make_pair(sorted_eigs, sorted_eigvecs);
  //auto secondpart = std::make_pair(atc,it);
  auto secondpart = std::make_pair(low_rank,it);
  return std::make_pair(firstpart, secondpart);
}

int main() {
  // Parameters for the adaptive algorithm
  double epsilon = 0.1; // perturbation size
  int deltapow = -5;
  int reltolpow = -2;//deltapow-2;
  int abstolpow = -4;//deltapow-4;
  double delta = std::pow(10,deltapow);  // Error
  double reltol = std::pow(10,reltolpow);
  double abstol = std::pow(10,abstolpow);

  unsigned numpanels = 100;
  unsigned m = MM;
  // Preparing the output files
  std::string fname = "kite_adapt_evecs";
  fname += "_" + to_string(m) + "_" + to_string(numpanels) + "_" + to_string(deltapow);
  std::ofstream out(fname);
  std::string fname1 = "kite_adapt_evals";
  fname1 += "_" + to_string(m) + "_" + to_string(numpanels) + "_" + to_string(deltapow);
  std::ofstream out1(fname1);
  std::string fname2 = "kite_adapt_its";
  fname2 += "_" + to_string(m) + "_" + to_string(numpanels) + "_" + to_string(deltapow);
  std::ofstream out2(fname2);
  std::string fname3 = "kite_adapt_evalex";
  fname3 += "_" + to_string(m) + "_" + to_string(numpanels) + "_" + to_string(deltapow);
  std::ofstream out3(fname3);

  // Definition of the domain
  static double R = 5;
  static double r = 0.5;

  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 3.5, 1.625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 3.5, 0;
  static parametricbem2d::ParametrizedFourierSum domain(
      Eigen::Vector2d(0, 0), cos_list, sin_list, 0, 2 * M_PI);

  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;
  unsigned q = space.getQ();

  // Definitions of the shape functional
  FF F(R, r);
  func f(R, r);

  unsigned dim = space.getSpaceDim(numpanels);
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  std::cout.precision(std::numeric_limits<double>::digits10);

  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  Eigen::MatrixXd A = bundle.first.first;  // Appears in quadratic term
  Eigen::MatrixXd M = bundle.first.second; // Defines Riemannian metric
  Eigen::VectorXd b = bundle.second;       // Appears in the linear term

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;

  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX M
  Eigen::MatrixXd L(M.llt().matrixL()); // Matrix L in L L^T
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim, dim);
  Eigen::MatrixXd Linv(L.lu().solve(Id));
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * b;                    // Linear part
  Eigen::MatrixXd A_tilde = Linv * A * Linv.transpose(); // Quadratic part

  // Getting the projection matrix
  Eigen::MatrixXd proj = Eigen::MatrixXd::Identity(dim, dim) -
                         b_tilde * b_tilde.transpose() / (b_tilde.dot(b_tilde));
  // Getting the matrix on which SVD is performed
  Eigen::MatrixXd A_svd = proj * A_tilde * proj;

  // Simple eigenvalue problem in transformed coordinates

  ges.compute(A_svd, Id); // Generalized solver

  // Testing
  // ges.compute(A, M);
  // Testing end

  // Absolute eigenvalues
  Eigen::VectorXd eigvals = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real();

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> templ(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(templ);

  ////////// Checking the subspace method
  Eigen::MatrixXd guess = Eigen::MatrixXd::Random(dim, dim);
  // Orthogonalization using QR decomposition in transformed coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(dim, dim);
  qr.compute(guess);
  Eigen::MatrixXd Q = qr.householderQ();
  // guess = Linv.transpose() * Q;
  guess = Q;

  bool atc_control_outer =
      true;             // If true, we use the additional termination criteria
  bool atc;

  unsigned K = 0; // Current dimension of subspace
  unsigned start_dim = K + 2;
  unsigned maxK = dim-2;
  // Matrix to capture the matrices of increasing sizes over iterations
  Eigen::MatrixXd container = Eigen::MatrixXd::Constant(dim, dim, 0);
  container.block(0, 0, dim, start_dim) = guess.block(0, 0, dim, start_dim);
  Eigen::VectorXd containervec = Eigen::VectorXd::Constant(dim, 0);
  Eigen::MatrixXd adapt_evals = Eigen::MatrixXd::Constant(dim,dim,0);
  Eigen::VectorXd adapt_its = Eigen::VectorXd::Constant(dim,0);

  double smallest_eig = delta / epsilon / epsilon + 1;

  // ADAPTIVE LOOP OVER K
  while (epsilon * epsilon * smallest_eig > delta && K < maxK) {
    ++K;
    std::cout << std::endl
              << "Iteration number (K) =  " << K << std::endl
              << std::endl;
    unsigned it_K = K + 2; // Dimension of slightly enlarged subspace
    Eigen::MatrixXd ortho_basis = container.block(0, 0, dim, it_K - 1);

    // Enlarging the matrix inside the container by one column
    Eigen::VectorXd randnewvec = Eigen::VectorXd::Random(dim);

    Eigen::VectorXd newvec = randnewvec - ortho_basis * ortho_basis.transpose() * Id * randnewvec;
    double normnewvec = sqrt(innerPdt(newvec,newvec,Id));
    newvec /= normnewvec;
    // container.col(it_K-1) = newvec - ortho_basis * ortho_basis.transpose() *
    // M * newvec;
    // Making the new vector orthogonal to all previous vectors
    container.col(it_K - 1) = newvec;

    Eigen::MatrixXd X0 = container.block(0, 0, dim, it_K);
    //Eigen::MatrixXd X0 = guess.block(0,0,dim,it_K); // Random initial guess

    // Subspace iteration method for eigensolution
    auto bund = SubspaceIteration(A_svd, Id, X0, reltol, abstol, delta);
    Eigen::MatrixXd evs = bund.first.second;
    Eigen::VectorXd evals = bund.first.first;
    containervec.segment(0, it_K) = evals;
    std::cout << "Found evals \n" << evals << std::endl;
    adapt_evals.col(K-1).segment(0,it_K) = evals;

    // Filling the container with computed eigenvectors
    container.block(0, 0, dim, it_K) = evs;
    smallest_eig = evals(K - 1);
    std::cout << "Smallest eigenvalue = " << smallest_eig << std::endl;

    // Additional termination criteria
    atc = bund.second.first;
    unsigned inner_its = bund.second.second;
    adapt_its(K-1) = inner_its;

    if (atc && atc_control_outer) {
      std::cout << "Using additional termination criteria to break outer loop!"
                << std::endl;
      K = it_K - 1;
      break;
    }

  } // ADAPTIVE LOOP ENDS

  // Getting the eigenvalues from the generalized eigen solver
  Eigen::VectorXd sorted_ges_eigs(dim);
  for (unsigned i = 0; i < dim; ++i) {
    sorted_ges_eigs(i) = eigvals(idx[i]);
  }
  Eigen::VectorXd sorted_ges_eigs_K = sorted_ges_eigs.segment(0, K);

  std::cout << "Error wrt Eigen ges: \n"
            << (sorted_ges_eigs_K - containervec.segment(0, K)).norm()
            << std::endl;
  out.precision(std::numeric_limits<double>::digits10);

  // Orthonormalizing the eigenvectors and bringing them back to the original
  // coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr1(dim, K);
  qr1.compute(container.block(0, 0, dim, K));
  Eigen::MatrixXd Q1 = qr.householderQ();

  // Transforming back to the original coordinate system
  // Eigen::MatrixXd output = Linv.transpose() * Q1;
  Eigen::MatrixXd output = Linv.transpose() * container.block(0, 0, dim, K);

  out << output << std::endl;

  out1.precision(std::numeric_limits<double>::digits10);
  if (atc)
    out1 << adapt_evals.block(0,0,K+1,K-1) << std::endl;
  else
    out1 << adapt_evals.block(0,0,K+2,K);


    if (atc)
      out2 << adapt_its.segment(0,K-1) << std::endl;
    else
      out2 << adapt_its.segment(0,K) << std::endl;
  //out1 << containervec.segment(0, K) << std::endl;

  std::cout << "All GES eigvals: \n" << sorted_ges_eigs << std::endl;
  out3.precision(std::numeric_limits<double>::digits10);
  out3 << sorted_ges_eigs << std::endl;
  std::cout << "Found subspace of dimension: " << K+1 << std::endl;

  return 0;
}
