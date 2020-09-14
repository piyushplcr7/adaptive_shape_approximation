#ifndef FUNCTIONALSHPP
#define FUNCTIONALSHPP

#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>
#include <iostream>

// f in \int_{\Omega} f dx. f = div F
class func {
private:
  double R_;
  double r_;

public:
  func(double RR, double rr) : R_(RR), r_(rr){};

  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);

#if MM == 1
    return 3 * x * x + 3 * y * y;
#endif

#if MM == 2
    return 3 * x * x + 2 * y;
#endif

#if MM == 3
    // std::cout << "Right f being used" << std::endl;
    return -std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) / r_ *
           (x + y - R_);
#endif

#if MM == 4
    double r = sqrt((x-R_) * (x-R_) + y * y);
    if (r <= 1)
      return 1 + cos(M_PI * r) - r * M_PI / 2. * sin(M_PI * r);
    else
      return 0;
#endif

#if MM == 5
    return 1 - fabs(x - R_) / r_;
#endif

#if MM == 6
    return 1 - (x - R_) / r_;
#endif

#if MM == 7
    if ((x - R_) * (x - R_) + y * y - r_ * r_ < 0)
      return std::pow((x - R_) * (x - R_) + y * y, 2) / 4. -
             ((x - R_) * (x - R_) + y * y) * r_ * r_ / 2.;
    else
      return 0;
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

#if MM == 3
    // std::cout << "Right gradf being used" << std::endl;
    return -std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) / r_ *
           Eigen::Vector2d(1 - (x + y - R_) * (x - R_) / r_,
                           1 - (x + y - R_) * y / r_);
#endif

#if MM == 4
    double r = sqrt((x-R_) * (x-R_) + y * y);
    if (r <= 1)
      return (-3 * M_PI / 2. * sin(M_PI * r) -
              r * M_PI * M_PI / 2. * cos(M_PI * r)) *
             Eigen::Vector2d((x-R_) / r, y / r);
    else
      return Eigen::Vector2d(0, 0);
#endif

#if MM == 5
    return Eigen::Vector2d((x - R_) > 0 ? -1 / r_ : 1 / r_, 0);
#endif

#if MM == 6
    return Eigen::Vector2d(-1 / r_, 0);
#endif

#if MM == 7
    if ((x - R_) * (x - R_) + y * y - r_ * r_ < 0)
      return ((x - R_) * (x - R_) + y * y - r_ * r_) *
             Eigen::Vector2d(x - R_, y);
    else
      return Eigen::Vector2d(0, 0);
#endif
  }
};

// F in \int_{\Gamma} F.n dS
class FF {
private:
  double R_;
  double r_;

public:
  FF(double RR, double rr) : R_(RR), r_(rr){};

  Eigen::Vector2d operator()(Eigen::Vector2d X) {
    double x = X(0);
    double y = X(1);

#if MM == 1
    return Eigen::Vector2d(std::pow(x, 3), std::pow(y, 3));
#endif

#if MM == 2
    return Eigen::Vector2d(x * x * x, y * y);
#endif

#if MM == 3
    // std::cout << "Right FF being used" << std::endl;
    return std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) *
           Eigen::Vector2d(1, 1);
#endif

#if MM == 4
    double r = sqrt((x-R_) * (x-R_) + y * y);
    double rho;

    if (r <= 1)
      rho = r * std::pow(cos(M_PI / 2. * r), 2);
    else
      rho = 0;

    return rho * Eigen::Vector2d((x-R_) / r, y / r);
#endif

#if MM == 5
    double Fx = (x - R_) - (x - R_) * fabs(x - R_) / 2. / r_;
    return Eigen::Vector2d(Fx, 0);
#endif

#if MM == 6
    double Fx = (x - R_) - (x - R_) * (x - R_) / 2. / r_;
    return Eigen::Vector2d(Fx, 0);
#endif
  }
};

/**
 * This Function takes a ParametrizedMesh object and outputs another
 * ParametrizedMesh object with the panels being the linear approximations of
 * the original ones.
 *
 * @param pmesh The mesh to be converted
 * @return The converted ParametrizedMesh
 */
parametricbem2d::ParametrizedMesh
convert_to_linear(const parametricbem2d::ParametrizedMesh &pmesh) {
  unsigned N = pmesh.getNumPanels();
  parametricbem2d::PanelVector lmesh;
  for (unsigned i = 0; i < N; ++i) {
    // Making linear panels using the end points of the original panel
    parametricbem2d::ParametrizedLine lpanel(pmesh.getVertex(i),
                                             pmesh.getVertex((i + 1) % N));
    parametricbem2d::PanelVector tmp = lpanel.split(1);
    lmesh.insert(lmesh.end(), tmp.begin(), tmp.end());
  }
  parametricbem2d::ParametrizedMesh plmesh(lmesh);
  return plmesh;
}

#endif
