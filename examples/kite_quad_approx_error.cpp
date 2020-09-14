#include "functionals.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include "functionals.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>

int main() {
  unsigned numpanels = 400;
  unsigned m = MM;
  // Preparing the output files
  std::string fname4 = "quadratic_approximation"; fname4 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out4(fname4);
  std::string fname1 = "full_functional.txt";
  std::ofstream out1(fname1);
  std::string fname2 = "error.txt";
  std::ofstream out2(fname2);

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
  FF F(R,r);

  func f(R, r);

  unsigned dim = space.getSpaceDim(numpanels);
  unsigned iterations = 2*dim;
  unsigned levels = 50;

  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));
  //parametricbem2d::ParametrizedMesh pmesh(domain.split(numpanels));
  //parametricbem2d::ParametrizedMesh mesh = convert_to_linear(pmesh);

  auto perturbed_shape_functional = [&](const Eigen::VectorXd &dcoeffs) {
    assert(dcoeffs.rows() == dim);
    double value = 0;
    parametricbem2d::PanelVector panels = mesh.getPanels();
    // Looping over panels to evaluate the perturbed functional
    for (unsigned i = 0; i < numpanels; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      // Velocity field over ith panel
      auto vel_field = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0; j < q; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j + 1, i + 1, mesh) - 1) *
                         space.evaluateShapeFunction(j, t);
        }
        // std::cout << "local field " << local_field << std::endl;
        return local_field;
      };
      // Velocity field derivative over ith panel
      auto vel_fieldp = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0; j < q; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j + 1, i + 1, mesh) - 1) *
                         space.evaluateShapeFunctionDot(j, t);
        }
        // std::cout << "local field prime " << local_field << std::endl;
        return local_field;
      };

      // Integrand over the ith panel
      auto Pi = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        Eigen::Vector2d ret = pi(t) + normal * vel_field(t);
        // std::cout << "ret pi " << ret << std::endl;
        return ret;
      };

      auto Pip = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        Eigen::Vector2d normdot =
            (tangent.norm() * pi.DoubleDerivative(t) -
             tangent * tangent.dot(pi.DoubleDerivative(t)) / tangent.norm()) /
            tangent.squaredNorm();

        Eigen::MatrixXd temp(2, 2);
        temp << 0, 1, -1, 0;
        normdot = temp * normdot;
        Eigen::Vector2d ret =
            pi.Derivative(t) + normal * vel_fieldp(t) + vel_field(t) * normdot;
        // std::cout << "ret pip " << ret << std::endl;
        return ret;
      };

      // Integrand restricted to ith panel
      auto integrand = [&](double t) {
        Eigen::Vector2d pt = Pi(t);
        Eigen::Vector2d tangent = Pip(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return F(pt).dot(normal) * tangent.norm();
      };
      value += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // std::cout << "after integral evaluation value updated to " << value <<
      // std::endl;
    }
    return value;
  };

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

  double SF0 = shape_functional(mesh);

  std::cout.precision(std::numeric_limits<double>::digits10);
  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  Eigen::MatrixXd A = bundle.first.first;
  Eigen::MatrixXd M = bundle.first.second;
  // Eigen::MatrixXd M = Eigen::MatrixXd::Identity(dim,dim);
  Eigen::VectorXd b = bundle.second;

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;

  ////////////////////////////////////////
  // ERICKS METHOD
  std::cout.precision(std::numeric_limits<double>::digits10);
  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX M
  Eigen::MatrixXd L(M.llt().matrixL()); // Matrix L in L L^T
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim, dim);
  //Eigen::MatrixXd Linv = L.inverse();
  Eigen::MatrixXd Linv(L.lu().solve(Id));
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * b;
  Eigen::MatrixXd A_tilde = Linv * A * Linv.transpose();
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

  // Finding the K dimensional subspace
  unsigned K = dim;
    Eigen::MatrixXd basis(dim, K);
    basis.col(0) = b_tilde;
    // Storing the top K eigenvectors
    for (unsigned i = 1; i < K; ++i) {
      basis.col(i) = eigvectorsl.col(idxl[i - 1]);
    }

    // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
    // Transforming the basis into original coordinates
    basis = Linv.transpose() * basis;
    //Eigen::MatrixXd ortho_basis = GramSchmidtOrtho(basis, M);
    Eigen::MatrixXd ortho_basis(basis);
    // Renormalizing
    for (unsigned i = 0 ; i < dim ; ++i) {
      Eigen::VectorXd col = ortho_basis.col(i);
      ortho_basis.col(i) /= sqrt(innerPdt(col,col,M) );
    }

  ////////////////////////////////////////

  Eigen::MatrixXd lrae = Eigen::VectorXd::Constant(levels, 0);
  Eigen::MatrixXd arae = Eigen::VectorXd::Constant(levels, 0);
  Eigen::MatrixXd err = Eigen::VectorXd::Constant(levels, 0);

  Eigen::VectorXd exponents = Eigen::VectorXd::LinSpaced(levels,-8,0);
  Eigen::VectorXd d0 = Eigen::VectorXd::Constant(dim,0);
  double pert0 = perturbed_shape_functional(d0);
  std::cout << "Pert0 check: " << pert0 << std::endl;
  std::cout << "SF0 check: " << SF0 << std::endl;

  for (int yo = 0; yo < levels; ++yo) {
    double dnorm = std::pow(10,exponents(yo));
    std::cout << "computation for perturbation of size: " << dnorm << std::endl;

    // Doing iterations to calculate supremum error by random sampling
    for (unsigned it = 0; it < iterations; ++it) {
      // Velocity field
      Eigen::VectorXd d(dim);

      if (it<dim) {
        d = ortho_basis.col(it); // Sampling from the basis
      }
      else { // Random sampling
        d = Eigen::VectorXd::Constant(dim, -1) + 2 * Eigen::VectorXd::Random(dim);
        d = ortho_basis * d; // Random linear combination of basis vectors
      }

      d *= dnorm / sqrt(innerPdt(d, d, M));

      double quad_approx = d.dot(A * d)/2. + b.dot(d) + SF0;

      double pert = perturbed_shape_functional(d);

      /*if (lrae(yo) < fabs((pert - quad_approx) / pert))
        lrae(yo) = fabs((pert - quad_approx) / pert);*/

      if (lrae(yo) < fabs((pert - quad_approx)))
        lrae(yo) = fabs((pert - quad_approx));


    } // Loop over it
  }   // Loop over yo

  out4.precision(std::numeric_limits<double>::digits10);
  out4 << lrae << std::endl;

  out1.precision(std::numeric_limits<double>::digits10);
  out1 << arae << std::endl;

  out2.precision(std::numeric_limits<double>::digits10);
  out2 << err << std::endl;

  return 0;
}
