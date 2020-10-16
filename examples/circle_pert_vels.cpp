#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>
#include "functionals.hpp"
#include "pert_vels.hpp"

int main() {
  // Number of panels
  unsigned numpanels = 400;
  // Number indicating the shape functional used
  unsigned m = MM;
  // Preparing the output files
  std::string fname2 = "circle_pert_vels"; fname2 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out2(fname2);
  std::string fname1 = "circle_eigvals"; fname1 += "_" + to_string(m)+ "_" + to_string(numpanels);
  std::ofstream out1(fname1);

  // Definition of the circular domain centered at (0,0)
  static double R = 1.5;
  static double r = 0.1;
  static parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0),
                                                         R, 0, 2 * M_PI);
  // Quadrature order
  unsigned order = 16;
  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;

  // Function f in \int_{\Omega} f(x) dx
  func f(R,r);

  // Dimension of the BEM space
  unsigned dim = space.getSpaceDim(numpanels);
  // Generating the parameterized mesh by splitting the domain into the required number of panels
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
