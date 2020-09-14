#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
  std::string fname = "quad_approx.txt";
  std::ofstream out(fname);
  double R = 1.5;
  // Definition of the domain
  parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0), R, 0,
                                                  2 * M_PI);

  // Definition using fourier sum param
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375, 0;
  parametricbem2d::ParametrizedFourierSum dom(Eigen::Vector2d(0, 0), cos_list,
                                              sin_list, 0, 2 * M_PI);
  // unsigned numpanels = 8;
  unsigned maxpanels = 100;
  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;

  // Definition of the shape functional
  auto F = [&](Eigen::Vector2d X) {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(x * x * x, y * y * y);
  };

  class func {
  public:
    double operator()(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return 3 * x * x + 3 * y * y;
    }

    Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return Eigen::Vector2d(6 * x, 6 * y);
    }
  };

  func f;

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

  /*unsigned rlevs = 20;
  Eigen::ArrayXd rs = Eigen::ArrayXd::LinSpaced(rlevs, R - 0.3, R + 0.3);
  Eigen::ArrayXd deltars = rs + (-R);
  Eigen::ArrayXd err = Eigen::ArrayXd::Constant(rlevs, 0);

  // Base level shape functional check
  double base_sf = shape_functional(mesh);
  double base_sf_ex = 1.5 * M_PI * std::pow(R, 4);
  std::cout << "shape functional : " << base_sf
            << " , exact formula: " << base_sf_ex << std::endl;
  for (unsigned i = 0; i < rlevs; ++i) {
    // Definition of the displacement vector field
    auto d = [&](const Eigen::Vector2d &X) {
      double x = X(0);
      double y = X(1);
      return deltars(i);
    };

    std::function<double(Eigen::Vector2d)> D = d;
    // Getting the shape gradient and hessian value for the displacement
    double shape_gradient = shapeGradient(mesh, f, D, order);
    double shape_hessian = shapeHessian(mesh, f, D, order);
    double quad_approx = base_sf_ex + shape_gradient + shape_hessian;
    double sf_ex = 1.5 * M_PI * std::pow(rs(i), 4);
    err(i) = fabs(sf_ex - quad_approx) / fabs(sf_ex);
  }
  Eigen::MatrixXd output(rlevs, 2);
  output << deltars, err;
  out.precision(std::numeric_limits<double>::digits10);
  out << std::setw(5) << "#deltars" << std::setw(15) << "error" << std::endl;
  out << output << std::endl;*/

  unsigned numpanels = 8;
  // Eigen::MatrixXd errmat = Eigen::MatrixXd::Constant(maxpanels, maxpanels,
  // 0);

  // for (unsigned numpanels = start; numpanels < maxpanels; numpanels += 3) {
  std::cout << "numpanels " << numpanels << std::endl;
  unsigned dim = space.getSpaceDim(numpanels);
  parametricbem2d::ParametrizedMesh mesh(dom.split(numpanels));
  // for (unsigned K = 1; K < dim; ++K) {
  // std::cout << "K: " << K << std::endl;
  // Do stuff
  Eigen::VectorXd d = Eigen::VectorXd::Random(dim);
  unsigned K = 3;
  double abs_err = compare(mesh, f, d, K, order, space);
  // errmat(numpanels, K) = abs_err;

  std::cout << "For K: " << K << ", abs_err = " << abs_err << std::endl;

  std::cout << "Trying the optimized way" << std::endl;

  auto bundle = getRelevantQuantities(mesh, f, order, space);

  Eigen::MatrixXd Mg = bundle.first.first;
  Eigen::MatrixXd A_pr = bundle.first.second;
  Eigen::VectorXd V = bundle.second;
  //std::tie(std::tie(Mg, A_pr), V) =
  //    getRelevantQuantities(mesh, f, d, order, space);

  Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Mg, A_pr);
  Eigen::VectorXd eigvals = ges.eigenvalues().real();     // eigenvalues
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(temp);
  Eigen::MatrixXd eigvectors_K(dim, K);
  // Storing the top K eigenvectors
  for (unsigned i = 0; i < K; ++i) {
    eigvectors_K.col(i) = eigvectors.col(idx[i]);
  }
  // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
  Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);
  Eigen::VectorXd dstar = Mg.lu().solve(V / 2.);
  double quad_approx =
      (d + dstar).dot(Mg * (d + dstar)) - dstar.dot(Mg * dstar);
  Eigen::VectorXd dpds_proj = ortho * ortho.transpose() * A_pr * (d + dstar);
  double low_rank_approx =
      dpds_proj.dot(Mg * dpds_proj) - dstar.dot(Mg * dstar);

  std::cout << "Quad approx " << quad_approx << " low rank " << low_rank_approx << std::endl;

  //  }
  //}
  // out << errmat << std::endl;

  return 0;
}
