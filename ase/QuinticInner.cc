#include <cmath>
#include <stdexcept>

#include "ase/QuinticInner.hh"

namespace ase {
    QuinticInner::QuinticInner(
        const double sp, const double yplus,
        const double sm, const double yminus)
        : sigmaPlus_(sp),
          yPlus_(yplus),
          sigmaMinus_(sm),
          yMinus_(yminus)
    {
        if (!(sigmaPlus_ > 0.0 && sigmaMinus_ > 0.0))
            throw std::invalid_argument(
                "In ase::QuinticInner constructor: "
                "both sigma arguments must be positive");

        const double sigsum = sm + sp;
        const double sprod = sm*sp;
        const double sm2 = sm*sm;
        const double sp2 = sp*sp;
        const double del = yminus - yplus;
        const double sigdel = sm - sp;
        const double denom = pow(sigsum, 5);
        const double tmp = 30*sprod*del/denom;

        double coeffs[6];
        coeffs[0] = ((10*sm2 + 5*sprod + sp2)*sp2*sp*yminus + sm2*sm*yplus*(sm2 + 5*sprod + 10*sp2))/denom;
        coeffs[1] = -tmp*sprod;
        coeffs[2] = tmp*sigdel;
        coeffs[3] = 10*(2*sprod - sigdel*sigdel)*del/denom;
        coeffs[4] = -15*del*sigdel/denom;
        coeffs[5] = -6*del/denom;

        quintic_ = Poly1D(coeffs, 5);
        deriv_ = quintic_.derivative();
        sder_ = deriv_.derivative();
    }

    double QuinticInner::operator()(const double x) const
    {
        if (x <= -sigmaMinus_)
            return yMinus_;
        else if (x < sigmaPlus_)
            return quintic_(x);
        else
            return yPlus_;
    }

    double QuinticInner::derivative(const double x) const
    {
        if (x <= -sigmaMinus_)
            return 0.0;
        else if (x < sigmaPlus_)
            return deriv_(x);
        else
            return 0.0;
    }

    double QuinticInner::secondDerivative(const double x) const
    {
        if (x <= -sigmaMinus_)
            return 0.0;
        else if (x < sigmaPlus_)
            return sder_(x);
        else
            return 0.0;
    }
}
