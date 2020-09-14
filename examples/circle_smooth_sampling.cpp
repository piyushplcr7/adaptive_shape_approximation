#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
  // Preparing the output files
  std::string fname = "circle_smooth_sampling.txt";
  std::ofstream out(fname);

  // Definition of the domain
  static double R = 1.5;
  static parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0),
                                                         R, 0, 2 * M_PI);

  static double c = 1; // Constant appearing in the velocity field
  std::cout << "c value : " << c << std::endl;

  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;

  // Definitions of the shape functional

  // F in \int_{\Gamma} F.n dS
  auto F = [&](Eigen::Vector2d X) {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d( std::pow(x,3), std::pow(y,3) );
    //return Eigen::Vector2d(x * x * x, y * y);
  };

  // f in \int_{\Omega} f dx. f = div F
  class func {
  public:
    double operator()(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return 3 * x * x + 3 * y * y;
      //return 3 * x * x + 2 * y;
    }

    Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return Eigen::Vector2d(6 * x, 6 * y);
      //return Eigen::Vector2d(6 * x, 2);
    }
  };

  func f;

  auto vel_field = [&](double x, double y) {
    //return c * cos(x) * cos(y);
    //return x * x * x * y * y;
    return c;
  };

  unsigned start = 4;
  unsigned maxpanels = 50; // start+1;

  Eigen::MatrixXd qa = Eigen::VectorXd::Constant(maxpanels, 0);
  Eigen::MatrixXd lra = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);
  Eigen::MatrixXd vfa = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);

  for (unsigned numpanels = start; numpanels < maxpanels; numpanels += 1) {
    std::cout << "numpanels " << numpanels << std::endl;
    unsigned dim = space.getSpaceDim(numpanels);
    parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

    Eigen::VectorXd d = space.Interpolate(vel_field, mesh);
    //std::cout << "Interpolated velocity field for c = " << c << " = \n" << d << std::endl;

    // Getting the relevant matrices
    auto bundle = getRelevantQuantities(mesh, f, order, space);
    Eigen::MatrixXd Mg = bundle.first.first;
    Eigen::MatrixXd A_pr = bundle.first.second;
    Eigen::VectorXd V = bundle.second;

    // Normalizing the velocity field d wrt the H1 norm
    double vnorm = sqrt(innerPdt(d, d, A_pr));

    // Calculating d*, which represents the linear part in the quadratic
    // functional
    Eigen::VectorXd dstar = Mg.lu().solve(V / 2.);

    // Calculating the QUADRATIC APPROXIMATION
    double quad_approx =
        (d + dstar).dot(Mg * (d + dstar)) - dstar.dot(Mg * dstar); // + SF0;
    qa(numpanels) = quad_approx;

    //std::cout << "quad_approx error: " << abs(quad_approx - perturbed_shape_functional) << std::endl;
    std::cout << "quad_approx: " << quad_approx << std::endl;
    // Solving the general eigenvalue problem
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(Mg, A_pr);
    Eigen::VectorXd eigvals = ges.eigenvalues().real();     // eigenvalues
    Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors

    // To check for NaNs in the eigensolution
    // std::cout << eigvectors.sum() << std::endl;

    // Storing the eigenvalues as std vector for sorting
    std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
    // Sorting while storing the indices
    auto idx = sort_indexes(temp);

    for (unsigned K = 1; K <= dim && false; ++K) {
      //std::cout << "K: " << K << std::endl;

      Eigen::MatrixXd eigvectors_K(dim, K);

      // Adding dstar to the approximation space
      if (true) {
        eigvectors_K.col(0) = dstar;

        // Storing the top K eigenvectors
        for (unsigned i = 1; i < K; ++i) {
          eigvectors_K.col(i) = eigvectors.col(idx[i - 1]);
        }

        // Check if inclusion of dstar causes linear dependence
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            eigvectors_K, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigvectors_K);

        auto rank = svd.rank();

        /*if (numpanels == maxpanels-1 && K <7) {
                  std::cout << eigvectors_K << std::endl;
                  std::cout << "rank: " << rank << std::endl;
            }*/

        if (rank < K) {
          std::cout << "Problem with including dstar, K = " << K
                    << " rank = " << rank << std::endl;
          Eigen::MatrixXd newmat(dim, K);
          // Storing the top K eigenvectors
          for (unsigned i = 0; i < K; ++i) {
            eigvectors_K.col(i) = eigvectors.col(idx[i]);
          }
        }
      }
      // Not adding dstar to the approximation space
      else {
        // Storing the top K eigenvectors
        for (unsigned i = 0; i < K; ++i) {
          eigvectors_K.col(i) = eigvectors.col(idx[i]);
        }
      }

      // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
      Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);

      // Low rank projection for d
      Eigen::VectorXd d_proj = ortho * ortho.transpose() * A_pr * d;

      // Low rank projection for dstar
      Eigen::VectorXd dstar_proj = ortho * ortho.transpose() * A_pr * dstar;

      // Low rank projection for d+d*
      Eigen::VectorXd dpds_proj = d_proj + dstar_proj;

      double low_rank_approx;
      // Calculating the low rank approximation in Low rank structure preserving
      // way
      if (true) {
        low_rank_approx =
            dpds_proj.dot(Mg * dpds_proj) - dstar.dot(Mg * dstar); // + SF0;
      }
      // Not low rank structure preserving way
      else {
        low_rank_approx = (d_proj + dstar).dot(Mg * (d_proj + dstar)) -
                          dstar.dot(Mg * dstar); // + SF0;
      }

      lra(numpanels, K) = low_rank_approx;

      vfa(numpanels, K) = (d + dstar - dpds_proj).norm();

      if (numpanels == maxpanels - 1 && K == dim - 1) {
        out2.precision(std::numeric_limits<double>::digits10);
        out2 << eigvectors_K << std::endl;
        for (unsigned i = 0; i < dim; ++i) {
          std::cout << eigvals(idx[i]) << " ,";
        }
        std::cout << std::endl;
      }
    } // Loop over K

    if (!true) {
      unsigned K = 3;
      double abs_err = compare(mesh, f, d, K, order, space);
      std::cout << "abs err: " << abs_err << std::endl;
    }

  } // Loop over numpanels
  out.precision(std::numeric_limits<double>::digits10);
  out << qa << std::endl;

  return 0;
}
