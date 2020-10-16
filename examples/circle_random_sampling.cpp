#include "functionals.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

int main() {
  unsigned numpanels = 400;
  unsigned m = MM;
  // Preparing the output files
  std::string fname = "circle_fae";
  fname += "_" + to_string(m) + "_" + to_string(numpanels);
  std::string fname1 = "circle_qae";
  fname1 += "_" + to_string(m) + "_" + to_string(numpanels);
  std::string fname4 = "circle_lrae";
  fname4 += "_" + to_string(m) + "_" + to_string(numpanels);
  std::ofstream out(fname);
  std::ofstream out1(fname1);
  std::ofstream out4(fname4);

  // Definition of the domain
  static double R = 1.5; // Radius of the circle
  static double r = 0.1; // Size of the support region, if applicable
  static parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0),
                                                         R, 0, 2 * M_PI);

  unsigned order = 16; // quadrature order

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;
  // Number of reference shape functions
  unsigned q = space.getQ();

  // Defines the vector field F. div(F) = f
  FF F(R, r);
  // Defines the function f appearing in the volume integral
  func f(R, r);

  // Dimensions of the space
  unsigned dim = space.getSpaceDim(numpanels);

  // Number of samples for calculating the errors. Half of the samples are for
  // calculating the errors at the basis vectors for the full obtained subspace.
  // The other half are for random linear combinations of these vectors
  unsigned iterations = 2 * dim;
  unsigned levels = 1;

  // Getting parametric mesh by splitting the domain
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  // This lambda function computes the perturbed shape functional \f$ J(\Omega_0
  // + \pmb{n} d) \f$ exactly. d is assumed to be a function in the BEM space
  // here.
  auto perturbed_shape_functional = [&](const Eigen::VectorXd &dcoeffs) {
    assert(dcoeffs.rows() == dim);
    double value = 0;
    // Getting the panels
    parametricbem2d::PanelVector panels = mesh.getPanels();
    // Looping over panels to evaluate the perturbed functional panelwise
    for (unsigned i = 0; i < numpanels; ++i) {
      // Current panel over which evaluation is done
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      // Velocity field over ith panel (scalar part)
      auto vel_field = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0; j < q; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j + 1, i + 1, mesh) - 1) *
                         space.evaluateShapeFunction(j, t);
        }
        return local_field;
      };
      // Velocity field's derivative over ith panel
      auto vel_fieldp = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0; j < q; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j + 1, i + 1, mesh) - 1) *
                         space.evaluateShapeFunctionDot(j, t);
        }
        return local_field;
      };

      // Parameterization for the perturbed panel
      auto Pi = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        // Finding the normal direction for normal perturbation
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        // Normal perturbation parameterization
        Eigen::Vector2d ret = pi(t) + normal * vel_field(t);
        return ret;
      };

      // Derivative of the parameterization for the perturbed panel
      auto Pip = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        // Finding the normal direction for normal perturbation
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        // The derivative of the parameterization for the perturbed panel
        Eigen::Vector2d normdot =
            (tangent.norm() * pi.DoubleDerivative(t) -
             tangent * tangent.dot(pi.DoubleDerivative(t)) / tangent.norm()) /
            tangent.squaredNorm();

        // Matrix used for calculations. This matrix gives a vector in the
        // outward normal direction when applied to the tangent vector
        Eigen::MatrixXd temp(2, 2);
        temp << 0, 1, -1, 0;
        normdot = temp * normdot;
        Eigen::Vector2d ret =
            pi.Derivative(t) + normal * vel_fieldp(t) + vel_field(t) * normdot;
        return ret;
      };

      // Integrand restricted to ith panel
      auto integrand = [&](double t) {
        // Getting the parameterization and its derivative
        Eigen::Vector2d pt = Pi(t);
        Eigen::Vector2d tangent = Pip(t);
        // Getting the normal vector
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        // The integrand \pmb{F} \cdot \pmb{n}
        return F(pt).dot(normal) * tangent.norm();
      };
      value += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
    }
    return value;
  };

  // This lambda function computes the shape functional J(\Omega_0)
  auto shape_functional = [&](const parametricbem2d::ParametrizedMesh &mesh) {

    unsigned numpanels = mesh.getNumPanels();
    parametricbem2d::PanelVector panels = mesh.getPanels();
    double val = 0;
    for (unsigned i = 0; i < numpanels; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return F(pi(s)).dot(normal) * pi.Derivative(s).norm();
      };
      val += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
    }
    return val;
  };

  double SF0 = shape_functional(mesh); // Shape functional for Omega_0

  std::cout.precision(std::numeric_limits<double>::digits10);

  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  Eigen::MatrixXd A = bundle.first.first;  // Appears in quadratic term
  Eigen::MatrixXd M = bundle.first.second; // Defines Riemannian metric
  Eigen::VectorXd b = bundle.second;       // Appears in the linear term

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;

  std::cout.precision(std::numeric_limits<double>::digits10);

  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX M
  // auto L(M.llt().matrixL());
  Eigen::MatrixXd L(M.llt().matrixL()); // Matrix L in L L^T
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim, dim);
  // Eigen::MatrixXd Linv = L.inverse(); // Might be problematic
  Eigen::MatrixXd Linv(L.lu().solve(Id));
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * b;                    // Linear part
  Eigen::MatrixXd A_tilde = Linv * A * Linv.transpose(); // Quadratic part

  // Getting the projection matrix
  Eigen::MatrixXd proj = Eigen::MatrixXd::Identity(dim, dim) -
                         b_tilde * b_tilde.transpose() / (b_tilde.dot(b_tilde));
  // Getting the matrix on which SVD is performed
  Eigen::MatrixXd A_svd = proj * A_tilde * proj;
  ges.compute(A_svd,
              Id); // Simple eigenvalue problem in transformed coordinates

  // Absolute eigenvalues
  Eigen::VectorXd eigvals = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real();

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> templ(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(templ);

  // Low rank approx. error
  Eigen::MatrixXd lrae = Eigen::MatrixXd::Constant(levels, numpanels, 0);
  // Quadratic approx. error
  Eigen::MatrixXd qae = Eigen::VectorXd::Constant(levels, 0);
  // Full approx. error
  Eigen::MatrixXd fae = Eigen::MatrixXd::Constant(levels, numpanels, 0);
  Eigen::VectorXd exponents = Eigen::VectorXd::LinSpaced(levels, -1, -1);

  // Constructing the basis with k = dim
  Eigen::MatrixXd basis_full(dim, dim);
  basis_full.col(0) = b_tilde;
  // Storing the top k-1 eigenvectors
  for (unsigned i = 1; i < dim; ++i) {
    basis_full.col(i) = eigvectors.col(idx[i - 1]);
  }

  // Orthogonalization using QR decomposition in transformed coordinate system
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(dim, dim);
  qr.compute(basis_full);
  Eigen::MatrixXd Q = qr.householderQ();

  // Transforming back to the original coordinate system
  basis_full = Linv.transpose() * Q;

  // Orthonormalizing
  // Eigen::MatrixXd ortho_basis_full = GramSchmidtOrtho(basis_full, M);

  Eigen::MatrixXd ortho_basis_full(basis_full);
  // Renormalizing
  for (unsigned i = 0; i < dim; ++i) {
    Eigen::VectorXd col = ortho_basis_full.col(i);
    ortho_basis_full.col(i) /= sqrt(innerPdt(col, col, M));
  }

  unsigned maxdim = 20;

  // Finding the K dimensional subspace
  for (unsigned K = 1; K <= maxdim; ++K) {
    std::cout << "K: " << K << std::endl;

    // Getting the k dimensional basis
    Eigen::MatrixXd ortho_basis = ortho_basis_full.block(0, 0, dim, K);

    for (int lev = 0; lev < levels; ++lev) {
      double dnorm = std::pow(10, exponents(lev));
      std::cout << "computation for perturbation of size: " << dnorm
                << std::endl;

      // Doing iterations to calculate supremum error by sampling
      for (unsigned it = 0; it < iterations; ++it) {
        // Velocity field
        Eigen::VectorXd d(dim);

        if (it < dim) {
          d = basis_full.col(it); // Sampling from the basis
        } else {                  // Random sampling
          d = Eigen::VectorXd::Constant(dim, -1) +
              2 * Eigen::VectorXd::Random(dim);
          d = basis_full * d; // Random linear combination of basis vectors
        }

        d *= dnorm / sqrt(innerPdt(d, d, M));

        // Calculating the QUADRATIC APPROXIMATION
        double quad_approx = d.dot(A * d) / 2. + b.dot(d) + SF0;

        // Projection of d using basis
        Eigen::VectorXd PKd = ortho_basis * ortho_basis.transpose() * M * d;

        // Calculating the low rank approximation
        double low_rank_approx = PKd.dot(A * PKd) / 2. + b.dot(PKd) + SF0;

        double pert = perturbed_shape_functional(d);

        if (lrae(lev, K - 1) <
            fabs((low_rank_approx - quad_approx))) // quad_approx))
          lrae(lev, K - 1) =
              fabs((low_rank_approx - quad_approx)); // quad_approx);

        if (qae(lev) < fabs((pert - quad_approx))) // pert))
          qae(lev) = fabs((pert - quad_approx));   // pert);

        if (fae(lev, K - 1) < fabs((pert - low_rank_approx))) // pert))
          fae(lev, K - 1) = fabs((pert - low_rank_approx));   // pert);
      }                                                       // Loop over it
    }                                                         // Loop over lev
  }                                                           // Loop over K

  // Error output
  out1.precision(std::numeric_limits<double>::digits10);
  out1 << qae << std::endl;

  out4.precision(std::numeric_limits<double>::digits10);
  out4 << lrae << std::endl;

  out.precision(std::numeric_limits<double>::digits10);
  out << fae << std::endl;

  /*Eigen::VectorXd d0 = Eigen::VectorXd::Constant(dim,1e-2);
  double pert0 = perturbed_shape_functional(d0);
  std::cout << "SF0: " << SF0 << std::endl;
  std::cout << "pert(0): " << pert0 << std::endl;*/

  return 0;
}
