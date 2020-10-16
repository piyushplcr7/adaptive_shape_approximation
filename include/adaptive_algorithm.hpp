#ifndef ADAPTIVEALGORITHMHPP
#define ADAPTIVEALGORITHMHPP

#include "functionals.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include "subspace_iteration.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

int main() {
  // Parameters for the adaptive algorithm
  double epsilon = 0.1; // perturbation size
  // exponents to define delta, reltol, abstol
  int deltapow = -4;
  int reltolpow = deltapow - 2;
  int abstolpow = deltapow - 4;
  // Defining the quantitites based on exponents
  double delta = std::pow(10, deltapow); // Error
  double reltol = std::pow(10, reltolpow);
  double abstol = std::pow(10, abstolpow);

  // Number of panels
  unsigned numpanels = 100;
  // Variable indicating the chosen functional
  unsigned m = MM;
  // Preparing the output files
  std::string fname = "circle_adapt_evecs";
  fname += "_" + to_string(m) + "_" + to_string(numpanels) + "_" +
           to_string(deltapow);
  std::ofstream out(fname);
  std::string fname1 = "circle_adapt_evals";
  fname1 += "_" + to_string(m) + "_" + to_string(numpanels) + "_" +
            to_string(deltapow);
  std::ofstream out1(fname1);
  std::string fname2 = "circle_adapt_its";
  fname2 += "_" + to_string(m) + "_" + to_string(numpanels) + "_" +
            to_string(deltapow);
  std::ofstream out2(fname2);

  // Definition of the circular domain centered at (0,0)
  static double R = 1.5;
  static double r = 0.1;
  static parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0),
                                                         R, 0, 2 * M_PI);

  // Quadrature order
  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;
  unsigned q = space.getQ();

  // Definitions for the shape functional
  // The vector field F such that f = div(F)
  FF F(R, r);
  // The sacalar function f appearing in \int_{\Omega} f(x) dx
  func f(R, r);

  // Getting the dimensions of the space
  unsigned dim = space.getSpaceDim(numpanels);
  // Splitting the domain into the required number of panels
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  // Setting precision for cout
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
  // Sorting and storing the indices
  auto idx = sort_indexes(templ);

  // Getting random vectors for initial guess
  Eigen::MatrixXd guess = Eigen::MatrixXd::Random(dim, dim);
  // Orthogonalization using QR decomposition in transformed coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(dim, dim);
  qr.compute(guess);
  Eigen::MatrixXd Q = qr.householderQ();
  // guess = Linv.transpose() * Q;
  guess = Q;

  bool atc_control_outer =
      true; // If true, we use the additional termination criteria
  bool atc;

  unsigned K = 0;             // Current dimension of subspace
  unsigned start_dim = K + 2; // Starting dimension for the enlarged subspace
  unsigned maxK = dim - 2;    // Maximum dimension for current subspace

  // Matrix to capture the matrices of increasing sizes over iterations
  Eigen::MatrixXd container = Eigen::MatrixXd::Constant(dim, dim, 0);
  // Filling the container with initial guess
  container.block(0, 0, dim, start_dim) = guess.block(0, 0, dim, start_dim);
  // Container of maximal size to hold the eigenvalues in the adaptive algorithm
  Eigen::VectorXd containervec = Eigen::VectorXd::Constant(dim, 0);
  // Container to store the eigenvalues obtained at each step of the adaptive
  // algorithm
  Eigen::MatrixXd adapt_evals = Eigen::MatrixXd::Constant(dim, dim, 0);
  // Container to store the iterations at each astep of the adaptive algorithm
  Eigen::VectorXd adapt_its = Eigen::VectorXd::Constant(dim, 0);

  // Initializing the smallest_eig variable to force the first iteration
  double smallest_eig = delta / epsilon / epsilon + 1;

  // ADAPTIVE LOOP OVER K
  while (epsilon * epsilon * smallest_eig > delta && K < maxK) {
    ++K;
    std::cout << std::endl
              << "Iteration number (K) =  " << K << std::endl
              << std::endl;
    unsigned it_K = K + 2; // Dimension of slightly enlarged subspace
    // Getting the orthonormal basis before enlarging the subspace
    Eigen::MatrixXd ortho_basis = container.block(0, 0, dim, it_K - 1);

    // Enlarging the matrix inside the container by one random column
    Eigen::VectorXd randnewvec = Eigen::VectorXd::Random(dim);
    // Making the new random vector orthogonal to the basis
    Eigen::VectorXd newvec =
        randnewvec - ortho_basis * ortho_basis.transpose() * Id * randnewvec;
    double normnewvec = sqrt(innerPdt(newvec, newvec, Id));
    // Normalizing the new random vector
    newvec /= normnewvec;
    // container.col(it_K-1) = newvec - ortho_basis * ortho_basis.transpose() *
    // M * newvec;
    container.col(it_K - 1) = newvec;

    // Extracting the initial guess from the big container of eigenvectors
    Eigen::MatrixXd X0 = container.block(0, 0, dim, it_K);
    // Eigen::MatrixXd X0 = guess.block(0,0,dim,it_K); // Random initial guess

    // Using the subspace iteration method for eigensolution
    auto bund = SubspaceIteration(A_svd, Id, X0, reltol, abstol, delta);
    // Extracting the eigenvectors
    Eigen::MatrixXd evs = bund.first.second;
    // Extracting the eigenvalues
    Eigen::VectorXd evals = bund.first.first;
    // Updating the big container for eigenvectors
    containervec.segment(0, it_K) = evals;
    std::cout << "Found evals \n" << evals << std::endl;
    // Updating the container for eigenvectors at each adaptive step
    adapt_evals.col(K - 1).segment(0, it_K) = evals;

    // Filling the big container with computed eigenvectors
    container.block(0, 0, dim, it_K) = evs;
    // Finding the smallest eigenvalue obtained at the current step
    smallest_eig = evals(K - 1);
    std::cout << "Smallest eigenvalue = " << smallest_eig << std::endl;

    // Additional termination criteria (Low rank)
    atc = bund.second.first;
    // Getting the number of iterations for the subspace iteration algorithm
    unsigned inner_its = bund.second.second;
    // Updating the container which stores iterations of the subspace iteration at each adaptive step
    adapt_its(K - 1) = inner_its;

    if (atc && atc_control_outer) {
      std::cout << "Using additional termination criteria to break outer loop!"
                << std::endl;
      // If low rank criteria used for breaking, updating K to point to the second last eigenvalue which is non zero.
      K = it_K - 1;
      break;
    }

  } // ADAPTIVE LOOP ENDS

  // Getting all the eigenvalues from the in built generalized eigen solver for verification
  Eigen::VectorXd sorted_ges_eigs(dim);
  for (unsigned i = 0; i < dim; ++i) {
    sorted_ges_eigs(i) = eigvals(idx[i]);
  }
  // Extracting the required number of eigenvalues
  Eigen::VectorXd sorted_ges_eigs_K = sorted_ges_eigs.segment(0, K);

  // Comparing the solution obtained with the adaptive algorithm against the
  // solution obtained from in built eigensolver
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
    out1 << adapt_evals.block(0, 0, K + 1, K - 1) << std::endl;
  else
    out1 << adapt_evals.block(0, 0, K + 2, K);

  if (atc)
    out2 << adapt_its.segment(0, K - 1) << std::endl;
  else
    out2 << adapt_its.segment(0, K) << std::endl;
  // out1 << containervec.segment(0, K) << std::endl;

  std::cout << "All GES eigvals: \n" << sorted_ges_eigs << std::endl;
  std::cout << "Found subspace of dimension: " << K + 1 << std::endl;

  return 0;
}

#endif
