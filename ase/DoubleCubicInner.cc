#include <stdexcept>

#include "ase/DoubleCubicInner.hh"

namespace ase {
    DoubleCubicInner::DoubleCubicInner(
        const double y0, const double sigmaPlus, const double yPlus,
        const double sigmaMinus, const double yMinus)
        : sigmaPlus_(sigmaPlus),
          yPlus_(yPlus),
          sigmaMinus_(sigmaMinus),
          yMinus_(yMinus),
          left_(y0, -sigmaMinus, yMinus),
          right_(y0, sigmaPlus, yPlus)
    {
        if (!(sigmaPlus > 0.0 && sigmaMinus > 0.0))
            throw std::invalid_argument(
                "In ase::DoubleCubicInner constructor: "
                "both sigma arguments must be positive");
    }

    double DoubleCubicInner::operator()(const double x) const
    {
        if (x <= -sigmaMinus_)
            return yMinus_;
        else if (x < 0.0)
            return left_(x);
        else if (x < sigmaPlus_)
            return right_(x);
        else
            return yPlus_;
    }

    double DoubleCubicInner::derivative(const double x) const
    {
        if (x <= -sigmaMinus_)
            return 0.0;
        else if (x < 0.0)
            return left_.derivative(x);
        else if (x < sigmaPlus_)
            return right_.derivative(x);
        else
            return 0.0;
    }

    double DoubleCubicInner::secondDerivative(const double x) const
    {
        if (x <= -sigmaMinus_)
            return 0.0;
        else if (x < 0.0)
            return left_.secondDerivative(x);
        else if (x < sigmaPlus_)
            return right_.secondDerivative(x);
        else
            return 0.0;
    }
}
