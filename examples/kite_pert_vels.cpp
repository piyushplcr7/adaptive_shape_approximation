#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "pert_vels.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>
#include "functionals.hpp"

int main() {
  // Number of panels
  unsigned numpanels = 400;
  unsigned m = MM;
  // Preparing the output files
  std::string fname2 = "kite_pert_vels"; fname2 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out2(fname2);
  std::string fname1 = "kite_eigvals"; fname1 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out1(fname1);

  // Definition of the kite domain
  static double R = 5;
  static double r = 0.5;

  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 3.5, 1.625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 3.5, 0;
  static parametricbem2d::ParametrizedFourierSum domain(
      Eigen::Vector2d(0, 0), cos_list, sin_list, 0, 2 * M_PI);

  // Quadrature order
  unsigned order = 16;
  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;

  // Function f in \int_{\Omega} f(x) dx
  func f(R,r);

  // Dimension of the BEM space
  unsigned dim = space.getSpaceDim(numpanels);
  // Generating the mesh by splitting domain into the required number of panels
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  // Matrix for getting the full subspace
  Eigen::MatrixXd ortho_basis;
  // Vector containing the eigenvalues
  Eigen::VectorXd sorted_eigvalsl;
  std::tie(ortho_basis, sorted_eigvalsl) = pert_vels(mesh, f, space, order);

  out2.precision(std::numeric_limits<double>::digits10);
  // Output of perturbation fields
  out2 << ortho_basis << std::endl;

  out1.precision(std::numeric_limits<double>::digits10);
  // Output of sorted eigenvalues
  out1 << sorted_eigvalsl << std::endl;

  return 0;
}
