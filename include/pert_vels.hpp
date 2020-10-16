#ifndef PERTVELSHPP
#define PERTVELSHPP

#include "abstract_bem_space.hpp"
#include "functionals.hpp"
#include "parametrized_mesh.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>

template <typename ff>
std::pair<Eigen::MatrixXd, Eigen::VectorXd>
pert_vels(const parametricbem2d::ParametrizedMesh &mesh, const ff &f,
          const parametricbem2d::AbstractBEMSpace &space, unsigned order) {
  // Number of panels
  unsigned numpanels = mesh.getNumPanels();

  // Dimension of the BEM space
  unsigned dim = space.getSpaceDim(numpanels);

  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  // Symmetric matrix for the quadratic part
  Eigen::MatrixXd A = bundle.first.first;
  // SPD matrix which defines the riemannian metric
  Eigen::MatrixXd M = bundle.first.second;
  // Vector which defines the linear part in quadratic functional
  Eigen::VectorXd b = bundle.second;

  // Solving the eigensystem
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  std::cout.precision(std::numeric_limits<double>::digits10);
  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX M
  Eigen::MatrixXd L(M.llt().matrixL()); // Matrix L in L L^T
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim, dim);
  // Eigen::MatrixXd Linv = L.inverse();
  Eigen::MatrixXd Linv(L.lu().solve(Id));
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * b;
  Eigen::MatrixXd A_tilde = Linv * A * Linv.transpose();
  // Getting the orthogonal complement matrix for SVD
  Eigen::MatrixXd rames =
      Eigen::MatrixXd::Identity(dim, dim) -
      b_tilde * b_tilde.transpose() / (b_tilde.dot(b_tilde));
  // Getting the projected matrix
  Eigen::MatrixXd A_svd = rames * A_tilde * rames;
  // Simple eigenvalue problem in transformed coordinates
  ges.compute(A_svd, Id);
  // Absolute eigenvalues
  Eigen::VectorXd eigvalsl = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectorsl = ges.eigenvectors().real(); // eigenvectors
  // Storing the eigenvalues as std vector for sorting
  std::vector<double> templ(eigvalsl.data(), eigvalsl.data() + dim);
  // Sorting while storing the indices
  auto idxl = sort_indexes(templ);

  // Computing the subspace of full dimension
  unsigned K = dim - 1;
  Eigen::MatrixXd basis_K(dim, K);

  // Adding b_tilde to the approximation space
  basis_K.col(0) = b_tilde;

  // Storing the top K eigenvectors
  for (unsigned i = 1; i < K; ++i) {
    basis_K.col(i) = eigvectorsl.col(idxl[i - 1]);
  }

  // Storing absolute eigenvalues in descending order
  Eigen::VectorXd sorted_eigvalsl(dim);
  for (unsigned i = 0; i < dim; ++i) {
    sorted_eigvalsl(i) = eigvalsl(idxl[i]);
    std::cout << eigvalsl(idxl[i]) << " , ";
  }

  // Orthogonalization using QR decomposition in transformed coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(dim, K);
  qr.compute(basis_K);
  Eigen::MatrixXd Q = qr.householderQ();

  // Transforming the basis into original coordinates
  basis_K = Linv.transpose() * Q;
  // Eigen::MatrixXd ortho_basis = GramSchmidtOrtho(basis_K, M);
  Eigen::MatrixXd ortho_basis(basis_K);
  // Renormalizing
  for (unsigned i = 0; i < dim - 1; ++i) {
    Eigen::VectorXd col = ortho_basis.col(i);
    // Normalizing
    ortho_basis.col(i) /= sqrt(innerPdt(col, col, M));
  }

  return std::make_pair(ortho_basis, sorted_eigvalsl);
}

#endif
