#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>

int main() {
  // Preparing the output files
  std::string fname = "quad_approx_circle.txt";
  std::string fname1 = "lowrank_approx_circle.txt";
  std::string fname2 = "eigvecs.txt";
  std::string fname3 = "velapprox.txt";
  std::string fname4 = "lowrank_approx_circlee.txt";
  std::ofstream out(fname);
  std::ofstream out1(fname1);
  std::ofstream out2(fname2);
  std::ofstream out3(fname3);
  std::ofstream out4(fname4);

  // Definition of the domain
  static double R = 1.5;
  static parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(R, 0),
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

    #if MM == 1
    return Eigen::Vector2d( std::pow(x,3), std::pow(y,3) );
    #endif

    #if MM == 2
    return Eigen::Vector2d(x * x * x, y * y);
    #endif
  };

  // f in \int_{\Omega} f dx. f = div F
  class func {
  public:
    double operator()(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);

      #if MM == 1
      return 3 * x * x + 3 * y * y;
      #endif

      #if MM == 2
      return 3 * x * x + 2 * y;
      #endif
    }

    Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);

      #if MM == 1
      return Eigen::Vector2d(6 * x, 6 * y);
      #endif

      #if MM == 2
      return Eigen::Vector2d(6 * x, 2);
      #endif
    }
  };

  func f;


  /*auto shape_functional = [&](const parametricbem2d::ParametrizedMesh &mesh) {
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

  class perturbation {
  public:
    // Constructor
    perturbation(std::shared_ptr<parametricbem2d::AbstractParametrizedCurve> pi)
        : pi_(pi) {}

    // Evaluation
    Eigen::Vector2d operator()(double t) const {
      Eigen::Vector2d point = pi_->operator()(t);
      Eigen::Vector2d tangent = pi_->Derivative(t);
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();

      // Cosine velocity field
      #if VEL == 1
      return point + c * cos(point(0)) * cos(point(1)) * normal;
      #endif

      // Constant velocity field
      #if VEL == 2
      return point + c * normal;
      #endif
    }

    Eigen::Vector2d Derivative(double t) const {
      Eigen::Vector2d pt = pi_->operator()(t);
      double x = pt(0);
      double y = pt(1);
      Eigen::Vector2d tangent = pi_->Derivative(t);
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();
      // Cosine velocity field

      #if VEL == 1
      return tangent +
             c * M_PI * Eigen::Vector2d(-sin((1 + t) * M_PI), cos((1 + t) * M_PI)) *
                 cos(x) * cos(y) +
             c * normal * M_PI *
                 (sin(x) * cos(y) * y - cos(x) * sin(y) * x);
      #endif

       // Constant velocity field
       #if VEL == 2
       return tangent + c * M_PI * Eigen::Vector2d(-sin((1 + t) * M_PI), cos((1 + t) * M_PI));
       #endif
    }

  private:
    // parametricbem2d::AbstractParametrizedCurve &pi_;
    std::shared_ptr<parametricbem2d::AbstractParametrizedCurve> pi_;
  };

  auto pshape_functional = [&](const perturbation &pert) {
    double val = 0;

    auto integrand = [&](double s) {
      Eigen::Vector2d pt = pert(s);
      Eigen::Vector2d tangent = pert.Derivative(s);
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();
      return F(pt).dot(normal) * tangent.norm();
    };
    val += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);

    return val;
  };*/

  auto vel_field = [&](double x, double y) {
    #if VEL == 1
    return c * cos(x) * cos(y);
    #endif

    //return x * x * x * y * y;
    #if VEL == 2
    return c;
    #endif
  };

  // Getting a shared pointer to the domain to initialize a perturbation object
  /*auto sp = std::make_shared<parametricbem2d::ParametrizedCircularArc>(
      Eigen::Vector2d(0, 0), R, 0, 2 * M_PI);
  perturbation pert(sp);

  double perturbed_shape_functional = pshape_functional(pert);

  std::cout << "J(Omega_0 + n d) (smooth): " << perturbed_shape_functional << std::endl;

  //const std::function<double(double, double)> d_cont = vel_field;
  auto d_cont = [&](Eigen::Vector2d X) {
    return vel_field(X(0),X(1));
  };*/

  unsigned start = 4;
  unsigned maxpanels = 50;
  //maxpanels = start+1;

  // Getting the exact shape functional at Omega_0
  /*parametricbem2d::ParametrizedMesh testmesh(domain.split(5));
  double ex_calc = shape_functional(testmesh);
  std::cout << "Shape functional at Omega_0 : " << ex_calc << std::endl<< std::endl;*/

  // Calculating the smooth shape gradient and hessian
  /*double sgs = shapeGradient(testmesh,f,d_cont,order);
  double shs = shapeHessian(testmesh, f, d_cont , order);
  std::cout << "Shape gradient (smooth): " << sgs << std::endl;
  std::cout << "Shape hessian (smooth): " << shs << std::endl;
  // Quadratic approximation without the constant term is sum of sg and sh
  std::cout << "Quadratic approximation (smooth): " << sgs+shs << std::endl << std::endl;*/

  // double SFt = 1.5 * M_PI * std::pow(R + t, 4);
  // std::cout << "SFt " << SFt << std::endl;

  Eigen::MatrixXd qa = Eigen::VectorXd::Constant(maxpanels, 0);
  Eigen::MatrixXd lra = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);
  Eigen::MatrixXd lrae = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);
  Eigen::MatrixXd vfa = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);

  for (unsigned numpanels = start; numpanels < maxpanels; numpanels += 1) {
    std::cout << "numpanels " << numpanels << std::endl;
    unsigned dim = space.getSpaceDim(numpanels);
    parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

    // Velocity field
    // Eigen::VectorXd d = t * Eigen::VectorXd::Constant(dim, 1);
    Eigen::VectorXd d = space.Interpolate(vel_field, mesh);
    //std::cout << "Interpolated velocity field for c = " << c << " = \n" << d << std::endl;

    // Getting the relevant matrices
    auto bundle = getRelevantQuantities(mesh, f, order, space);
    Eigen::MatrixXd Mg = bundle.first.first;
    Eigen::MatrixXd A_pr = bundle.first.second;
    Eigen::VectorXd V = bundle.second;

    // Normalizing the velocity field d wrt the H1 norm
    //d /= sqrt(innerPdt(d, d, A_pr));

    // Calculating d* representing the linear part in the quadratic functional
    Eigen::VectorXd dstar = Mg.lu().solve(V / 2.);

    // Calculating the QUADRATIC APPROXIMATION
    double quad_approx =
        (d + dstar).dot(Mg * (d + dstar)) - dstar.dot(Mg * dstar); // + SF0;
    qa(numpanels) = quad_approx;

    //std::cout << "Shape gradient (discrete): " << shapeGradient(mesh,f,d,order,space) << std::endl;
    //std::cout << "Shape Hessian (discrete): " << shapeHessian(mesh,f,d,order,space) << std::endl;
    //std::cout << "quad_approx error: " << abs(quad_approx - perturbed_shape_functional) << std::endl;
    //std::cout << "quad_approx: " << quad_approx << std::endl;

    // Solving the general eigenvalue problem RALF's METHOD
    // Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    //A_pr = Eigen::MatrixXd::Identity(dim,dim);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(Mg, A_pr);
    // Absolute eigenvalues
    Eigen::VectorXd eigvals = ges.eigenvalues().real().cwiseAbs();
    Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors

    // Storing the eigenvalues as std vector for sorting
    std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
    // Sorting while storing the indices
    auto idx = sort_indexes(temp);

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
    //std::cout << "eigvalsl" << std::endl << eigvalsl << std::endl;
    Eigen::MatrixXd eigvectorsl = ges.eigenvectors().real(); // eigenvectors
    //std::cout << "eigvecsl" << std::endl << eigvectorsl << std::endl;

    //std::cout << "Basic check from matlab" << std::endl << V.transpose() * A_pr * eigvectorsl << std::endl;

    // Storing the eigenvalues as std vector for sorting
    std::vector<double> templ(eigvalsl.data(), eigvalsl.data() + dim);
    // Sorting while storing the indices
    auto idxl = sort_indexes(templ);

    // To check for NaNs in the eigensolution
    // std::cout << eigvectors.sum() << std::endl;

    // Finding the K dimensional subspace
    for (unsigned K = 1; K <= dim; ++K) {
      std::cout << "K: " << K << std::endl;

      Eigen::MatrixXd eigvectors_K(dim, K);
      Eigen::MatrixXd basis_K(dim,K);

      // Adding dstar to the approximation space
      if (true) {
        eigvectors_K.col(0) = dstar;
        basis_K.col(0) = b_tilde;

        // Storing the top K eigenvectors
        for (unsigned i = 1; i < K; ++i) {
          eigvectors_K.col(i) = eigvectors.col(idx[i - 1]);
          basis_K.col(i) = eigvectorsl.col(idxl[i - 1]);
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
      // Transforming the basis into original coordinates
      basis_K = Linv.transpose() * basis_K;
      Eigen::MatrixXd ortho_basis = GramSchmidtOrtho(basis_K, A_pr);

      /*if (K == 2) {
        std::cout << "Basis_K check" << std::endl << basis_K.transpose() * A_pr * basis_K << std::endl;
        std::cout << "orthogonalization check" << std::endl << ortho_basis.transpose() * A_pr * ortho_basis << std::endl;
      }*/

      // Low rank projection for d
      Eigen::VectorXd d_proj = ortho * ortho.transpose() * A_pr * d;
      // Projection using Ericks basis
      Eigen::VectorXd PKd = ortho_basis * ortho_basis.transpose() * A_pr * d;

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

      double low_rank_approx_erick = PKd.dot(Mg * PKd) + V.dot(PKd) ;
      // V is just b from x^T A x + b^T x, d <-> x, A_pr <-> M
      std::cout << "b orthogonal to PKd-d: " << fabs(V.transpose()*A_pr*(PKd-d)) << std::endl;
      std::cout << "Error for linear term: " << fabs(V.transpose()*(PKd-d)) << std::endl;

      lra(numpanels, K) = low_rank_approx;
      lrae(numpanels, K) = low_rank_approx_erick;
      vfa(numpanels, K) = (d + dstar - dpds_proj).norm();

      if (numpanels == maxpanels - 1 && K == dim - 1) {
        out2.precision(std::numeric_limits<double>::digits10);
        //out2 << eigvectors_K << std::endl;
        //out2 << basis_K << std::endl;
        out2 << ortho_basis << std::endl;
        /*for (unsigned i = 0; i < dim; ++i) {
          std::cout << eigvals(idx[i]) << " ,";
        }
        std::cout << std::endl;*/
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
  out1.precision(std::numeric_limits<double>::digits10);
  out1 << lra << std::endl;
  out3.precision(std::numeric_limits<double>::digits10);
  out3 << vfa << std::endl;
  out4.precision(std::numeric_limits<double>::digits10);
  out4 << lrae << std::endl;

  return 0;
}
