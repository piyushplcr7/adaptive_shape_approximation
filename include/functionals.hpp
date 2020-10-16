#ifndef FUNCTIONALSHPP
#define FUNCTIONALSHPP

#include <Eigen/Dense>
#include <iostream>

/**
 * \class func
 * \brief This class provides different implementations for the scalar valued
 * function \f$ f := \nabla \cdot \pmb{F} \f$ appearing in the domain integral
 * shape functional : \f$ J(\Omega) := \int_{\Omega} f(\pmb{x}) d\pmb{x} \f$ and
 * its gradient \f$ \nabla f \f$ used in shape derivatives.
 */
class func {
private:
  // Parameter for center of function f
  double R_;
  // Parameter of radius of function f
  double r_;

public:
  func(double RR, double rr) : R_(RR), r_(rr){};

  // Evaluation operator
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);

    // Return values for different functions
#if MM == 1
    return 3 * x * x + 3 * y * y;
#endif

#if MM == 2
    return 3 * x * x + 2 * y;
#endif

#if MM == 3
    return -std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) / r_ *
           (x + y - R_);
#endif

#if MM == 4
    double r = sqrt((x - R_) * (x - R_) + y * y);
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

  // Gradient operator for the function f
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
    return -std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) / r_ *
           Eigen::Vector2d(1 - (x + y - R_) * (x - R_) / r_,
                           1 - (x + y - R_) * y / r_);
#endif

#if MM == 4
    double r = sqrt((x - R_) * (x - R_) + y * y);
    if (r <= 1)
      return (-3 * M_PI / 2. * sin(M_PI * r) -
              r * M_PI * M_PI / 2. * cos(M_PI * r)) *
             Eigen::Vector2d((x - R_) / r, y / r);
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

/**
 * \class FF
 * \brief This class provides different implementations for the vector valued
 * function \f$ \pmb{F} \f$ which allows us to write the domain integral shape
 * functional in boundary integral form \f$ J(\Omega) = \int_{\partial\Omega}
 * \pmb{F}\cdot \pmb{n} dS \f$. \f$ \nabla \cdot \pmb{F} = f \f$.
 */
class FF {
private:
  // Private fields for center and radius of the function f = div(F)
  double R_;
  double r_;

public:
  FF(double RR, double rr) : R_(RR), r_(rr){};

  // Evaluation operator
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
    return std::exp(-0.5 / r_ * ((x - R_) * (x - R_) + y * y)) *
           Eigen::Vector2d(1, 1);
#endif

#if MM == 4
    double r = sqrt((x - R_) * (x - R_) + y * y);
    double rho;

    if (r <= 1)
      rho = r * std::pow(cos(M_PI / 2. * r), 2);
    else
      rho = 0;

    return rho * Eigen::Vector2d((x - R_) / r, y / r);
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

#endif
