#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include "functionals.hpp"

int main() {
  // Preparing the output files
  //std::string fname = "quad_approx_circle.txt";
  //std::string fname1 = "lowrank_approx_circle.txt";
  std::string fname4 = "ellipse_random_sampling5.txt";
  //std::ofstream out(fname);
  //std::ofstream out1(fname1);
  std::ofstream out4(fname4);

  // Definition of the domain
  static double r = 0.1;
  static double R = 3;

  Eigen::MatrixXd cos_list(2, 1);
  cos_list << R, 0;
  Eigen::MatrixXd sin_list(2, 1);
  sin_list << 0, 1.5;
  static parametricbem2d::ParametrizedFourierSum domain(
      Eigen::Vector2d(0, 0), cos_list, sin_list, 0, 2 * M_PI);

  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;
  unsigned q = space.getQ();

  // Definitions of the shape functional

  // F in \int_{\Gamma} F.n dS
  auto F = [&](Eigen::Vector2d X) {
    double x = X(0);
    double y = X(1);

    #if MM == 1
    return Eigen::Vector2d( std::pow(x,3), std::pow(y,3) );
    #endif

    #if MM == 2
    return Eigen::Vector2d(x * x * x, y * y);
    #endif

    return Eigen::Vector2d(x * x * x, y * y);
  };

  func f(R,r);

  unsigned numpanels = 100;
  unsigned iterations = 100;
  unsigned levels = 11;

  unsigned dim = space.getSpaceDim(numpanels);
  parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

  auto perturbed_shape_functional = [&](const Eigen::VectorXd& dcoeffs) {
    assert(dcoeffs.cols() == dim);
    double value = 0;
    parametricbem2d::PanelVector panels = mesh.getPanels();
    // Looping over panels to evaluate the perturbed functional
    for (unsigned i = 0 ; i < numpanels ; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      // Velocity field over ith panel
      auto vel_field = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0 ; j < q ; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j+1,i+1,mesh)-1) *
                          space.evaluateShapeFunction(j,t);
        }
        return local_field;
      };
      // Velocity field derivative over ith panel
      auto vel_fieldp = [&](double t) {
        double local_field = 0;
        for (unsigned j = 0 ; j < q ; ++j) {
          local_field += dcoeffs(space.LocGlobMap2(j+1,i+1,mesh)-1) *
                          space.evaluateShapeFunctionDot(j,t);
        }
        return local_field;
      };

      // Integrand over the ith panel
      auto Pi = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return pi(t) + normal * vel_field(t);
      };

      auto Pip = [&](double t) {
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        Eigen::Vector2d normdot = (tangent.norm() * pi.DoubleDerivative(t) -
                                  tangent * tangent.dot(pi.DoubleDerivative(t))
                                  /tangent.norm() )/tangent.squaredNorm();

        Eigen::MatrixXd temp(2,2);
        temp << 0,1,-1,0;
        normdot = temp * normdot;
        return pi.Derivative(t) + normal * vel_fieldp(t) + vel_field(t) * normdot;
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
      value += parametricbem2d::ComputeIntegral(integrand,-1,1,order);
    }
    return value;
  };

  std::cout.precision(std::numeric_limits<double>::digits10);
  // Getting the relevant matrices
  auto bundle = getRelevantQuantities(mesh, f, order, space);
  Eigen::MatrixXd Mg = bundle.first.first;
  Eigen::MatrixXd A_pr = bundle.first.second;
  //Eigen::MatrixXd A_pr = Eigen::MatrixXd::Identity(dim,dim);
  Eigen::VectorXd V = bundle.second;
  // Calculating d* representing the linear part in the quadratic functional
  //Eigen::VectorXd dstar = Mg.colPivHouseholderQr().solve(V / 2.);

  // Solving the general eigenvalue problem RALF's METHOD
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  /*ges.compute(Mg, A_pr);
  // Absolute eigenvalues
  Eigen::VectorXd eigvals = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
  // Sorting while storing the indices
  auto idx = sort_indexes(temp);*/

  // ERICKS METHOD
  std::cout.precision(std::numeric_limits<double>::digits10);
  // CHOLESKY DECOMPOSITION OF THE RIEMANNIAN MATRIX A_pr
  Eigen::MatrixXd L(A_pr.llt().matrixL() ); // Matrix L in L L^T
  Eigen::MatrixXd Linv = L.inverse();
  // Getting the transformed matrix and vector in the quadratic functional
  Eigen::VectorXd b_tilde = Linv * V;
  Eigen::MatrixXd A_tilde = Linv * Mg * Linv.transpose();
  // Getting the orthogonal complement matrix for SVD
  Eigen::MatrixXd rames = Eigen::MatrixXd::Identity(dim,dim) - b_tilde * b_tilde.transpose() / (b_tilde.dot(b_tilde));
  Eigen::MatrixXd A_svd = rames * A_tilde * rames;
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim,dim);
  ges.compute(A_svd, Id);
  // Absolute eigenvalues
  Eigen::VectorXd eigvalsl = ges.eigenvalues().real().cwiseAbs();
  Eigen::MatrixXd eigvectorsl = ges.eigenvectors().real(); // eigenvectors

  // Erick's method for identity
  /*Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(dim,dim);
  Eigen::MatrixXd rames = Id - V * V.transpose()/ (V.dot(V));
  Eigen::MatrixXd A_svd = rames * Mg * rames;
  std::cout << "Al\n" << A_svd << std::endl;
  ges.compute(A_svd,Id);
  // Absolute eigenvalues
  Eigen::VectorXd eigvalsl = ges.eigenvalues().real().cwiseAbs();
  std::cout << "Eigvalsl\n" << eigvalsl << std::endl;
  Eigen::MatrixXd eigvectorsl = ges.eigenvectors().real(); // eigenvectors*/

  // Storing the eigenvalues as std vector for sorting
  std::vector<double> templ(eigvalsl.data(), eigvalsl.data() + dim);
  // Sorting while storing the indices
  auto idxl = sort_indexes(templ);

  //Eigen::MatrixXd lra = Eigen::MatrixXd::Constant(levels, numpanels, 0);
  Eigen::MatrixXd lrae = Eigen::MatrixXd::Constant(levels, numpanels, 0);

    // Finding the K dimensional subspace
    for (unsigned K = 1; K <= dim; ++K) {
      std::cout << "K: " << K << std::endl;

      //Eigen::MatrixXd eigvectors_K(dim, K);
      Eigen::MatrixXd basis_K(dim,K);

      // Adding dstar to the approximation space
        //eigvectors_K.col(0) = dstar;
        basis_K.col(0) = b_tilde;

        // Storing the top K eigenvectors
        for (unsigned i = 1; i < K; ++i) {
          //eigvectors_K.col(i) = eigvectors.col(idx[i - 1]);
          basis_K.col(i) = eigvectorsl.col(idxl[i - 1]);
        }

        // Check if inclusion of dstar causes linear dependence
        /*Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            eigvectors_K, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto rank = svd.rank();

        if (rank < K) {
          std::cout << "Problem with including dstar, K = " << K
                    << " rank = " << rank << std::endl;
          Eigen::MatrixXd newmat(dim, K);
          // Storing the top K eigenvectors
          for (unsigned i = 0; i < K; ++i) {
            eigvectors_K.col(i) = eigvectors.col(idx[i]);
          }
        }*/

        // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
        //Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);
        // Transforming the basis into original coordinates
        basis_K = Linv.transpose() * basis_K;
        Eigen::MatrixXd ortho_basis = GramSchmidtOrtho(basis_K, A_pr);

        for (int yo = 0 ; yo<levels ; ++yo) {
          int eps = yo-levels+1;

          double dnorm = std::pow(10,eps);
          std::cout << "computation for perturbation of size: " << dnorm << std::endl;

      // Doing iterations to calculate supremum error by random sampling
      for (unsigned it = 0 ; it < iterations ; ++it) {
        // Velocity field
        Eigen::VectorXd d = Eigen::VectorXd::Constant(dim,-1)+ 2 * Eigen::VectorXd::Random(dim);
        // Setting the correct norm for the perturbation field
        d *= dnorm/sqrt(innerPdt(d, d, A_pr));

        // Calculating the QUADRATIC APPROXIMATION
        //double quad_approx =
        //    (d + dstar).dot(Mg * (d + dstar)) - dstar.dot(Mg * dstar); // + SF0;
        double quad_approx = d.dot(Mg*d) + V.dot(d);

        // Low rank projection for d
        //Eigen::VectorXd d_proj = ortho * ortho.transpose() * A_pr * d;
        // Projection using Ericks basis
        Eigen::VectorXd PKd = ortho_basis * ortho_basis.transpose() * A_pr * d;

        // Low rank projection for dstar
        //Eigen::VectorXd dstar_proj = ortho * ortho.transpose() * A_pr * dstar;

        // Low rank projection for d+d*
        //Eigen::VectorXd dpds_proj = d_proj + dstar_proj;

        //double low_rank_approx;
        // Calculating the low rank approximation in Low rank structure preserving
        // way
        /*if (true) {
          low_rank_approx =
              dpds_proj.dot(Mg * dpds_proj) - dstar.dot(Mg * dstar); // + SF0;
        }
        // Not low rank structure preserving way
        else {
          low_rank_approx = (d_proj + dstar).dot(Mg * (d_proj + dstar)) -
                            dstar.dot(Mg * dstar); // + SF0;
        }*/

        double low_rank_approx_erick = PKd.dot(Mg * PKd) + V.dot(PKd) ;

        /*if (lra(yo,K-1) < fabs((low_rank_approx-quad_approx)/quad_approx))
          lra(yo, K-1) = fabs((low_rank_approx-quad_approx)/quad_approx);*/

        if (lrae(yo, K-1) < fabs((low_rank_approx_erick-quad_approx)/quad_approx))
        lrae(yo, K-1) = fabs((low_rank_approx_erick-quad_approx)/quad_approx);
      } // Loop over it
    } // Loop over yo
    } // Loop over K

  /*out1.precision(std::numeric_limits<double>::digits10);
  out1 << lra << std::endl;*/
  out4.precision(std::numeric_limits<double>::digits10);
  out4 << lrae << std::endl;

  return 0;
}
