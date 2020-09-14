#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>
#include "functionals.hpp"

int main() {
  unsigned numpanels = 400;
  unsigned m = MM;
  // Preparing the output files
  std::string fname2 = "kite_pert_vels"; fname2 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out2(fname2);
  std::string fname1 = "kite_eigvals"; fname1 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out1(fname1);

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

  func f(R,r);

  unsigned dim = space.getSpaceDim(numpanels);
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  Eigen::MatrixXd Mg = bundle.first.first;
  Eigen::MatrixXd A_pr = bundle.first.second;
  Eigen::VectorXd V = bundle.second;

  // Debugging
  /*std::cout << "Mg\n" << Mg << std::endl;
  std::cout << "A_pr\n" << A_pr << std::endl;
  std::cout << "V\n" << V << std::endl;*/

  // Calculating d* representing the linear part in the quadratic functional
  /*Eigen::VectorXd dstar = Mg.lu().solve(V / 2.);

  // Solving the general eigenvalue problem RALF's METHOD
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Mg, A_pr);
  // Absolute eigenvalues
  Eigen::VectorXd eigvals = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors
  // Storing the eigenvalues as std vector for sorting
  std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(temp);*/

  // ERICKS METHOD
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  std::cout.precision(std::numeric_limits<double>::digits10);
  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX A_pr
  Eigen::MatrixXd L(A_pr.llt().matrixL()); // Matrix L in L L^T
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim, dim);
  //Eigen::MatrixXd Linv = L.inverse();
  Eigen::MatrixXd Linv(L.lu().solve(Id));
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * V;
  Eigen::MatrixXd A_tilde = Linv * Mg * Linv.transpose();
  // Getting the orthogonal complement matrix for SVD
  Eigen::MatrixXd rames =
      Eigen::MatrixXd::Identity(dim, dim) -
      b_tilde * b_tilde.transpose() / (b_tilde.dot(b_tilde));
  Eigen::MatrixXd A_svd = rames * A_tilde * rames;
  ges.compute(A_svd, Id);
  // Absolute eigenvalues
  Eigen::VectorXd eigvalsl = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectorsl = ges.eigenvectors().real(); // eigenvectors
  // Storing the eigenvalues as std vector for sorting
  std::vector<double> templ(eigvalsl.data(), eigvalsl.data() + dim);
  // Sorting while storing the indices
  auto idxl = sort_indexes(templ);

  unsigned K = dim - 1;
  //Eigen::MatrixXd eigvectors_K(dim, K); // Ralf's method
  Eigen::MatrixXd basis_K(dim, K); // Erick's method

  // Adding dstar to the approximation space (Ralf)
  //eigvectors_K.col(0) = dstar;
  // Adding b_tilde to the approximation space (Erick)
  basis_K.col(0) = b_tilde;

  // Storing the top K eigenvectors
  for (unsigned i = 1; i < K; ++i) {
    //eigvectors_K.col(i) = eigvectors.col(idx[i - 1]);
    basis_K.col(i) = eigvectorsl.col(idxl[i - 1]);
  }

  Eigen::VectorXd sorted_eigvalsl(dim);
  for (unsigned i = 0 ; i < dim ; ++i) {
    sorted_eigvalsl(i) = eigvalsl(idxl[i]);
    std::cout << eigvalsl(idxl[i]) << " , ";
  }

  // Orthogonalization using QR decomposition in transformed coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(dim,K);
  qr.compute(basis_K);
  Eigen::MatrixXd Q = qr.householderQ();

  // Transforming the basis into original coordinates
  basis_K = Linv.transpose() * Q;
  //Eigen::MatrixXd ortho_basis = GramSchmidtOrtho(basis_K, A_pr);
  Eigen::MatrixXd ortho_basis(basis_K);
  // Renormalizing
  for (unsigned i = 0 ; i < dim-1 ; ++i) {
    Eigen::VectorXd col = ortho_basis.col(i);
    ortho_basis.col(i) /= sqrt(innerPdt(col,col,A_pr) );
  }

  out2.precision(std::numeric_limits<double>::digits10);
  // Output of perturbation fields given by Erick's method
  out2 << ortho_basis << std::endl;

  out1.precision(std::numeric_limits<double>::digits10);
  // Output of perturbation fields given by Erick's method
  out1 << sorted_eigvalsl << std::endl;

  return 0;
}
