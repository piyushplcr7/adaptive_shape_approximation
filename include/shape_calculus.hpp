#ifndef SHAPECALCULUSHPP
#define SHAPECALCULUSHPP

#include "abstract_bem_space.hpp"
#include "continuous_space.hpp"
#include "gauleg.hpp"
#include "integral_gauss.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

/*
 * This function outputs the sort indexes after a sort operation
 *
 * @param v Vector which is to be sorted
 *
 * @return vector which contains the sorted indices
 */
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

/*
 * This function calculates the inner product between vec1 and vec2 based on the
 * Matrix A: vec1^T A vec2
 *
 * @param vec1 The first vector
 * @param vec2 The second vector
 * @param A The matrix which defines the inner product
 *
 * @return the inner product defined as vec1^T A vec2
 */
double innerPdt(const Eigen::VectorXd &vec1, const Eigen::VectorXd &vec2,
                const Eigen::MatrixXd &A) {
  return vec1.dot(A * vec2);
}

/*
 * This function performs Gram Schmidt Process on the columns of the matrix
 * elems based on the inner product a^T A b for two vectors a and b. The output
 * matrix contains normalized columns.
 *
 * WARNING: INEFFICIENT IMPLEMENTATION
 *
 * @param elems The matrix whose columns are to be orthonormalized
 * @param A The matrix which defines the inner product between columns of elems
 *
 * @return Matrix with the same dimensions as elems, with orthonormalized
 * columns
 */
Eigen::MatrixXd GramSchmidtOrtho(const Eigen::MatrixXd &elems,
                                 const Eigen::MatrixXd &A) {
  unsigned k = elems.cols();
  Eigen::MatrixXd ortho = elems;
  for (unsigned i = 0; i < k; ++i) {
    for (unsigned j = 0; j < i; ++j) {
      // orthogonalization step
      ortho.col(i) -= innerPdt(elems.col(i), ortho.col(j), A) /
                      innerPdt(ortho.col(j), ortho.col(j), A) * ortho.col(j);
    }
    // Normalization step
    ortho.col(i) /= sqrt(innerPdt(ortho.col(i), ortho.col(i), A));
  }
  return ortho;
}

/*
 * This function computes the exact shape gradient for the shape functional :
 * \f$ \int_{\Gamma} f(x) dx = \int_{\Gamma} \operatorname{div}F(x) dx \f$. The
 * shape gradient formula is based on normal-perturbation model where the
 * function d specifies the magnitude of normal perturbation on the boundary.
 *
 * @tparam F template parameter for f. Should support evaluation operator.
 * @param mesh Parametric mesh object which defines the reference shape \f$
 * \Omega_0 \f$
 * @param f input of type F
 * @param d Perturbation field
 * @param order Order for numerical quadrature
 *
 * @return The shape gradient
 */
template <typename F>
double shapeGradient(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                     const std::function<double(Eigen::Vector2d)> &d,
                     unsigned order) {
  // Initializing
  double shape_gradient = 0;
  // Getting information about the panels
  unsigned numpanels = mesh.getNumPanels();
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Evaluating the shape gradient formula panelwise
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Shape gradient formula, pulled back under parameterization pi
    auto integrand = [&](double s) {
      return d(pi(s)) * f(pi(s)) * pi.Derivative(s).norm();
    };
    // Adding up individual contributions
    shape_gradient += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
  }
  return shape_gradient;
}

/*
 * This function computes the exact shape Hessian for the shape functional :
 * \f$ \int_{\Gamma} f(x) dx = \int_{\Gamma} \operatorname{div}F(x) dx \f$. The
 * shape gradient evaluation is based on normal-perturbation model where the
 * function d specifies the magnitude of normal perturbation on the boundary.
 *
 * @tparam F template parameter for f. Should support evaluation operator and a
 * 'grad' method for evaluating the gradient.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Perturbation field
 * @param order Order for numerical quadrature
 *
 * @return The shape hessian
 */
template <typename F>
double shapeHessian(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                    const std::function<double(Eigen::Vector2d)> &d,
                    unsigned order) {
  // Initializing
  double shape_hessian = 0;
  // Getting info about the mesh panels
  unsigned numpanels = mesh.getNumPanels();
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Evaluating the shape hessian panelwise
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Shape hessian formula, pulled back under the parameterization pi
    auto integrand = [&](double s) {
      Eigen::Vector2d tangent = pi.Derivative(s);
      // Finding the outward normal (assuming anti-clockwise curve)
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();
      Eigen::MatrixXd M(2, 2);
      M << pi.Derivative(s), pi.DoubleDerivative(s);
      // Curvature of the boundary
      double kappa = M.determinant() / std::pow(pi.Derivative(s).norm(), 3);
      return std::pow(d(pi(s)), 2) *
             (f.grad(pi(s)).dot(normal) + kappa * f(pi(s))) *
             pi.Derivative(s).norm();
    };
    // Adding up the individual contributions
    shape_hessian += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
  }
  return shape_hessian;
}

/*
 * This function computes the shape gradient for the domain integral shape
 * functional : \f$ \int_{\Omega} f(x) dx = \int_{\Omega} \operatorname{div}F(x)
 * dx \f$. The shape gradient formula is based on the normal perturbation model
 * where the normal perturbations are modulated by the function d on the
 * boundary. For this function, d lies in a Boundary Element Space and specified
 * through two arguments: Vector of coefficients and the type of BEM space
 *
 *
 * @tparam F template parameter for f. Should support evaluation operator.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Coefficients for the perturbation field lying in BEM space
 * @param order Order for numerical quadrature
 * @param space BEM space in which the function d lies
 *
 * @return The shape gradient
 */
template <typename F>
double shapeGradient(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                     const Eigen::VectorXd &d, unsigned order,
                     const parametricbem2d::AbstractBEMSpace &space) {
  // Getting the panels and their total number
  unsigned numpanels = mesh.getNumPanels();
  parametricbem2d::PanelVector panels = mesh.getPanels();

  // Getting BEM space info
  // Number of reference shape functions
  unsigned q = space.getQ();
  // Dimension of the space
  unsigned dim = space.getSpaceDim(numpanels);
  if (d.rows() != dim) {
    throw std::runtime_error(
        "Velocity field vector not valid! Please check the dimensions.");
  }

  // Initializing a vector which contains integrals for basis functions
  Eigen::VectorXd V = Eigen::VectorXd::Constant(dim, 0);

  // Looping over all panels
  for (unsigned i = 0; i < numpanels; ++i) {
    // Panel for local integral
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Looping over all RSFs
    for (unsigned k = 0; k < q; ++k) {
      // Lambda function for the local integrand
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) * f(pi(s)) *
               pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double loc_integ =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);

      // Local to global mapping
      unsigned I = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      V(I) += loc_integ;
    }
  }
  // Summing contributions from all the basis functions
  return d.dot(V);
}

/*
 * This function computes the exact shape Hessian for the shape functional :
 * \f$ \int_{\Gamma} f(x) dx = \int_{\Gamma} \operatorname{div}F(x) dx \f$. The
 * shape gradient evaluation is based on the normal perturbation model
 * where the normal perturbations are modulated by the function d on the
 * boundary. For this function, d is assumed to lie in a Boundary Element Space
 * which is provided as an argument along with the vector of coefficients.
 *
 * @tparam F template parameter for f. Should support evaluation operator and a
 * 'grad' method for evaluating the gradient.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Coefficients for the perturbation field lying in BEM space
 * @param order Order for numerical quadrature
 * @param space BEM space in which the function d lies
 *
 * @return The shape hessian
 */
template <typename F>
double shapeHessian(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                    const Eigen::VectorXd &d, unsigned order,
                    const parametricbem2d::AbstractBEMSpace &space) {
  // Getting the panels and their total number
  unsigned numpanels = mesh.getNumPanels();
  parametricbem2d::PanelVector panels = mesh.getPanels();

  // Getting BEM space info
  // Number of reference shape functions
  unsigned q = space.getQ();
  // Dimension of the space
  unsigned dim = space.getSpaceDim(numpanels);

  // Initializing a matrix for integrals of pairs of basis functions
  Eigen::MatrixXd M = Eigen::MatrixXd::Constant(dim, dim, 0);

  // Looping over all panels
  for (unsigned i = 0; i < numpanels; ++i) {
    // Panel for local integral
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Looping over RSFs
    for (unsigned k = 0; k < q; ++k) {
      for (unsigned l = 0; l < q; ++l) {
        // Lambda function for the local integrand
        auto integrand = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          // Getting the outward normal
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          Eigen::MatrixXd M(2, 2);
          M << pi.Derivative(s), pi.DoubleDerivative(s);
          // The curvature of the boundary
          double kappa = M.determinant() / std::pow(pi.Derivative(s).norm(), 3);
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) *
                 (f.grad(pi(s)).dot(normal) + kappa * f(pi(s))) *
                 pi.Derivative(s).norm();
        };
        // Evaluating the local integral
        double loc_int =
            parametricbem2d::ComputeIntegral(integrand, -1, 1, order);

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        unsigned JJ = space.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        M(II, JJ) += loc_int;
      } // Loop over l
    }   // Loop over k
  }     // Loop over i
  return d.dot(M * d);
}

template <typename F>
Eigen::VectorXd calc_d_star(const parametricbem2d::ParametrizedMesh &mesh,
                            const F &f, const Eigen::VectorXd &d,
                            unsigned order,
                            const parametricbem2d::AbstractBEMSpace &space) {
  // Getting the panels and their total number
  unsigned numpanels = mesh.getNumPanels();
  parametricbem2d::PanelVector panels = mesh.getPanels();

  // Defining the BEM space
  // parametricbem2d::ContinuousSpace<1> space;
  unsigned q = space.getQ();
  unsigned dim = space.getSpaceDim(numpanels);

  // Initializing the matrix for integrals of the basis functions
  Eigen::MatrixXd M = Eigen::MatrixXd::Constant(dim, dim, 0);
  // Initializing the vector of basis function integrals
  Eigen::VectorXd V = Eigen::VectorXd::Constant(dim, 0);

  // Looping over all panels
  for (unsigned i = 0; i < numpanels; ++i) {
    // Panel for local integral
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    for (unsigned k = 0; k < q; ++k) {
      auto integrand_V = [&](double s) {
        return space.evaluateShapeFunction(k, s) * f(pi(s)) *
               pi.Derivative(s).norm();
      };

      for (unsigned l = 0; l < q; ++l) {
        // Lambda function for the local integrand
        auto integrand_M = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          Eigen::MatrixXd M(2, 2);
          M << pi.Derivative(s), pi.DoubleDerivative(s);
          double kappa = M.determinant() / std::pow(pi.Derivative(s).norm(), 3);
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) *
                 (f.grad(pi(s)).dot(normal) + kappa * f(pi(s))) *
                 pi.Derivative(s).norm();
        };

        double loc_int_M =
            parametricbem2d::ComputeIntegral(integrand_M, -1, 1, order);

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        unsigned JJ = space.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        M(II, JJ) += loc_int_M;
      } // Loop over l
      double loc_integ =
          parametricbem2d::ComputeIntegral(integrand_V, -1, 1, order);

      // Local to global mapping
      unsigned I = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      //  std::cout << "Position " << I << " in V is filled" << std::endl;
      V(I) += loc_integ;
    } // Loop over k
  }   // Loop over i
  return M.lu().solve(V / 2.);
}

/*
 * This function solves the discrete eigenvalue problem:
 * \f$ \int_{\Gamma_0} g2 u v dS = \lambda <u,v>_{H_1(\Gamma_0)} \quad \forall v
 * \in V \f$ and returns the coefficients of the projected velocity field onto
 * the span of top K eigenvectors.
 *
 * @tparam F template parameter for f. Should support evaluation operator and a
 * 'grad' method for evaluating the gradient.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Perturbation field
 * @param K The dimension of the subspace to which d is projected
 * @param order Order for numerical quadrature
 * @param space BEM space in which the perturbation field lies
 *
 * @return The projected velocity field
 */
template <typename F>
Eigen::VectorXd getProjected(const parametricbem2d::ParametrizedMesh &mesh,
                             const F &f, const Eigen::VectorXd &d, unsigned K,
                             unsigned order,
                             const parametricbem2d::AbstractBEMSpace &space) {
  // BEM space for trial and test function
  // Getting number of panels in the mesh
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dim = space.getSpaceDim(numpanels);
  // Number of reference shape functions in the space
  unsigned q = space.getQ();
  // Initializing the LHS Matrix M_g2
  Eigen::MatrixXd Mg = Eigen::MatrixXd::Constant(dim, dim, 0);
  // Initializing the RHS matrix A
  Eigen::MatrixXd A = Mg;
  // Initializing the L^2 norm matrix
  Eigen::MatrixXd M = A;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();

  // Looping over all the panels
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];

    // Going over the reference shape functions
    for (unsigned k = 0; k < q; ++k) {
      for (unsigned l = 0; l < q; ++l) {
        // Lambda function for g2(pi(s)) = g(s)
        auto g = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          Eigen::MatrixXd M(2, 2);
          M << pi.Derivative(s), pi.DoubleDerivative(s);
          double kappa = M.determinant() / std::pow(pi.Derivative(s).norm(), 3);
          return f.grad(pi(s)).dot(normal) + kappa * f(pi(s));
        };

        // local integrand for LHS
        auto integrand_lhs = [&](double s) {
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) * g(s) *
                 pi.Derivative(s).norm();
        };

        // local integrand for H1 seminorm term, surface gradients -> shapefndot
        auto integrand_h1 = [&](double s) {
          return space.evaluateShapeFunctionDot(k, s) *
                 space.evaluateShapeFunctionDot(l, s) / pi.Derivative(s).norm();
        };

        // local integrand for L^2 norm
        auto integrand_l2 = [&](double s) {
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) * pi.Derivative(s).norm();
        };

        // Evaluating the local integrals
        double integral_lhs =
            parametricbem2d::ComputeIntegral(integrand_lhs, -1, 1, order);
        double integral_h1 =
            parametricbem2d::ComputeIntegral(integrand_h1, -1, 1, order);
        double integral_l2 =
            parametricbem2d::ComputeIntegral(integrand_l2, -1, 1, order);

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        unsigned JJ = space.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        // Filling the global matrices at the right place
        Mg(II, JJ) += integral_lhs;
        A(II, JJ) += integral_h1;
        M(II, JJ) += integral_l2;
      } // loop over l ends
    }   // loop over k ends
  }     // loop over i ends

  // Getting the matrix for the total H1 norm
  Eigen::MatrixXd A_pr = A + M;
  // std::cout.precision(std::numeric_limits<double>::digits10);
  std::cout << "Mg \n" << Mg << std::endl;
  std::cout << "A_pr \n " << A_pr << std::endl;
  std::cout << "Determinants:  Mg      A      M       A_pr" << std::endl;
  std::cout << Mg.determinant() << " " << A.determinant() << " "
            << M.determinant() << " " << A_pr.determinant() << std::endl;

  // Solving the generalized eigenvalue problem Mg X = \lambda A' X
  // Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Mg, A_pr);
  Eigen::VectorXd eigvals = ges.eigenvalues().real();     // eigenvalues
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors
  std::cout << "eigvals \n" << eigvals << std::endl;
  std::cout << "eigvecs \n" << eigvectors << std::endl;

  Eigen::MatrixXd wasakamana = Eigen::MatrixXd::Constant(dim, dim, 0);
  wasakamana.diagonal() = eigvals;
  // Verifying the eigensolution
  std::cout << "Verification \n"
            << Mg * eigvectors - A_pr * eigvectors * wasakamana << std::endl;

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(temp);
  Eigen::MatrixXd eigvectors_K(dim, K);
  // Storing the top K eigenvectors
  for (unsigned i = 0; i < K; ++i) {
    eigvectors_K.col(i) = eigvectors.col(idx[i]);
  }

  // Checking if the K chosen eigenvectors are linearly independent
  // Doing SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigvectors_K, Eigen::ComputeThinU |
                                                          Eigen::ComputeThinV);
  auto rank = svd.rank();
  std::cout << "Rank : " << rank << std::endl;
  // Rank of full eigvector matrix
  Eigen::JacobiSVD<Eigen::MatrixXd> svdfull(
      eigvectors, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto rankf = svdfull.rank();
  std::cout << "Rank full : " << rankf << std::endl;

  // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
  Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);

  std::cout << "orthogonalized matrix \n" << ortho << std::endl;

  Eigen::MatrixXd testmat1(K, K);
  for (unsigned i = 0; i < K; ++i)
    for (unsigned j = 0; j < K; ++j)
      testmat1(i, j) = innerPdt(eigvectors_K.col(i), eigvectors_K.col(j), A_pr);

  Eigen::MatrixXd testmat2(K, K);
  for (unsigned i = 0; i < K; ++i)
    for (unsigned j = 0; j < K; ++j)
      testmat2(i, j) = innerPdt(ortho.col(i), ortho.col(j), A_pr);

  std::cout << "testmat1 \n" << testmat1 << std::endl;
  std::cout << "testmat2 \n" << testmat2 << std::endl;

  // Projection of the velocity field onto the span of orthonormalized K
  // eigenvectors
  return ortho * ortho.transpose() * A_pr * d;
}

template <typename F>
Eigen::MatrixXd getEigVectors(const Eigen::MatrixXd &Mg,
                              const Eigen::MatrixXd &A_pr) {
  // Solving the generalized eigenvalue problem Mg X = \lambda A' X
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Mg, A_pr);
  Eigen::VectorXd eigvals = ges.eigenvalues().real();     // eigenvalues
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors
  return eigvectors;
}

Eigen::VectorXd getProjected(const Eigen::MatrixXd &eigvectors_K,
                             const Eigen::MatrixXd &A_pr,
                             const Eigen::VectorXd &d) {
  Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);
  return ortho * ortho.transpose() * A_pr * d;
}

template <typename F>
std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd>
getRelevantQuantities(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                      unsigned order,
                      const parametricbem2d::AbstractBEMSpace &space) {
  // BEM space for trial and test function
  // Getting number of panels in the mesh
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dim = space.getSpaceDim(numpanels);
  // Number of reference shape functions in the space
  unsigned q = space.getQ();
  // Initializing the LHS Matrix M_g2 - Corresponds to the quadratic term in the
  // quaddratic approximation functional wrt velocity basis
  Eigen::MatrixXd Mg = Eigen::MatrixXd::Constant(dim, dim, 0);
  // Initializing the RHS matrix A - Corresponds to the H1 seminorm for velocity
  // basis
  Eigen::MatrixXd A = Mg;
  // Initializing the L^2 norm matrix
  Eigen::MatrixXd M = A;
  // Initializing the vector of basis function integrals - Corresponds to the
  // linear term in the quadratic approximation functional wrt the velocity
  // basis
  Eigen::VectorXd V = Eigen::VectorXd::Constant(dim, 0);
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();

  // Looping over all the panels
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Going over the reference shape functions
    for (unsigned k = 0; k < q; ++k) {
      auto integrand_V = [&](double s) {
        return space.evaluateShapeFunction(k, s) * f(pi(s)) *
               pi.Derivative(s).norm();
      };

      for (unsigned l = 0; l < q; ++l) {
        // Lambda function for g2(pi(s)) = g(s)
        auto g = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          Eigen::MatrixXd M(2, 2);
          M << pi.Derivative(s), pi.DoubleDerivative(s);
          double kappa = M.determinant() / std::pow(pi.Derivative(s).norm(), 3);
          // std::cout << "Kappa = " << kappa << std::endl;
          return f.grad(pi(s)).dot(normal) + kappa * f(pi(s));
        };

        // local integrand for LHS
        auto integrand_lhs = [&](double s) {
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) * g(s) *
                 pi.Derivative(s).norm();
        };

        // local integrand for H1 seminorm term, surface gradients -> shapefndot
        auto integrand_h1 = [&](double s) {
          return space.evaluateShapeFunctionDot(k, s) *
                 space.evaluateShapeFunctionDot(l, s) / pi.Derivative(s).norm();
        };

        // local integrand for L^2 norm
        auto integrand_l2 = [&](double s) {
          return space.evaluateShapeFunction(k, s) *
                 space.evaluateShapeFunction(l, s) * pi.Derivative(s).norm();
        };

        // Evaluating the local integrals
        double integral_lhs =
            parametricbem2d::ComputeIntegral(integrand_lhs, -1, 1, order);

        // std::cout << "(shape hessian) Loc integ for panel " << i << " : " <<
        // integral_lhs << std::endl;
        double integral_h1 =
            parametricbem2d::ComputeIntegral(integrand_h1, -1, 1, order);
        double integral_l2 =
            parametricbem2d::ComputeIntegral(integrand_l2, -1, 1, order);

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        unsigned JJ = space.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        // Filling the global matrices at the right place
        Mg(II, JJ) += integral_lhs;
        A(II, JJ) += integral_h1;
        M(II, JJ) += integral_l2;
      } // loop over l ends
      double loc_integ =
          parametricbem2d::ComputeIntegral(integrand_V, -1, 1, order);
      // std::cout << "Local integral for the shape gradient term: " <<
      // loc_integ << std::endl; std::cout << "(shape gradient) Loc integ for
      // panel " << i << " : " << loc_integ << std::endl;
      // Local to global mapping
      unsigned I = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      V(I) += loc_integ;
    } // loop over k ends
  }   // loop over i ends

  // Getting the matrix for the total H1 norm
  Eigen::MatrixXd A_pr = A + M;

  return std::make_pair(std::make_pair(Mg, A_pr), V);
}

/*
 * Returns the discrete quadratic approximation without the constant term: shape
 * gradient + shape hessian. The perturbation field d is assumed to lie in a BEM
 * space provided in the arguments.
 *
 * @tparam F template parameter for f. Should support evaluation operator and a
 * 'grad' method for evaluating the gradient.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Vector of coefficients for the perturbation field
 * @param order Order for numerical quadrature
 * @param space BEM space in which the perturbation field lies
 *
 * @return quadratic approximation without the constant term
 */
template <typename F>
double quadApprox(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
                  const Eigen::VectorXd &d, unsigned order,
                  const parametricbem2d::AbstractBEMSpace &space) {
  double t1 = shapeGradient(mesh, f, d, order, space);
  double t2 = shapeHessian(mesh, f, d, order, space);
  return t1 + t2;
}

/*
 * Compares the discrete quadratic approximation with its low rank approximation
 *
 * @tparam F template parameter for f. Should support evaluation operator and a
 * 'grad' method for evaluating the gradient.
 * @param mesh Parametric mesh object which defines \f$ \Omega_0 \f$
 * @param f input of type F
 * @param d Perturbation field
 * @param K The dimension of the subspace to which d is projected
 * @param order Order for numerical quadrature
 * @param space BEM space in which the perturbation field lies
 *
 * @return absolute error for the low rank approximation
 */
template <typename F>
double compare(const parametricbem2d::ParametrizedMesh &mesh, const F &f,
               const Eigen::VectorXd &d, unsigned K, unsigned order,
               const parametricbem2d::AbstractBEMSpace &space) {
  // Computing the discrete quadratic approximation, without the constant term
  double quad_approx = quadApprox(mesh, f, d, order, space);
  // Computing the low rank approximation for the quadratic approximation
  // Getting the projected velocity field
  // Eigen::VectorXd d_projected = getProjected(mesh, f, d, K, order, space);
  // Evaluating the low rank approximation based on the projected field
  // double low_rank_approx = quadApprox(mesh, f, d_projected, order, space);
  // std::cout << "Coefficients for original field \n" << d << std::endl;
  // std::cout << "Coefficients for projected field \n"
  //          << d_projected << std::endl;
  Eigen::VectorXd dstar = calc_d_star(mesh, f, d, order, space);
  std::cout << "dstar: \n" << dstar << std::endl;
  std::cout << "quad approx: " << quad_approx << " quad approx sq compl: "
            << shapeHessian(mesh, f, d + dstar, order, space) -
                   shapeHessian(mesh, f, dstar, order, space)
            << std::endl;
  /*std::cout << "quad approx: " << quad_approx
            << " low rank: " << low_rank_approx << " quad approx sq compl: "
            << shapeHessian(mesh, f, d + dstar, order, space) -
                   shapeHessian(mesh, f, dstar, order, space)
            << std::endl;*/
  Eigen::VectorXd d_plus_dstar_projected =
      getProjected(mesh, f, d + dstar, K, order, space);
  std::cout << "dpds proj \n " << d_plus_dstar_projected << std::endl;
  double real_low_rank =
      shapeHessian(mesh, f, d_plus_dstar_projected, order, space) -
      shapeHessian(mesh, f, dstar, order, space);
  std::cout << "Low rank approximation (square completion) error: "
            << fabs(
                   shapeHessian(mesh, f, d_plus_dstar_projected, order, space) -
                   shapeHessian(mesh, f, dstar, order, space) - quad_approx)
            << std::endl;
  // Calculating the absolute error
  // return fabs(quad_approx - low_rank_approx);
  return fabs(quad_approx - real_low_rank);
}

template <typename F>
double low_rank_approx(const parametricbem2d::ParametrizedMesh &mesh,
                       const F &f, const Eigen::VectorXd &d, unsigned K,
                       unsigned order,
                       const parametricbem2d::AbstractBEMSpace &space) {

  Eigen::VectorXd dstar = calc_d_star(mesh, f, d, order, space);
  Eigen::VectorXd d_plus_dstar_projected =
      getProjected(mesh, f, d + dstar, K, order, space);

  double real_low_rank =
      shapeHessian(mesh, f, d_plus_dstar_projected, order, space) -
      shapeHessian(mesh, f, dstar, order, space);
  return real_low_rank;
}

#endif
