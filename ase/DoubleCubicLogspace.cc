#include <cmath>
#include <cassert>

#include "ase/DoubleCubicLogspace.hh"

namespace ase {
    DoubleCubicLogspace::DoubleCubicLogspace(
        const double y0, const double sigmaPlus, const double yPlus,
        const double sigmaMinus, const double yMinus)
        : inner_(log(y0), sigmaPlus, log(yPlus), sigmaMinus, log(yMinus))
    {
        assert(y0 > 0.0);
        assert(yPlus > 0.0);
        assert(yMinus > 0.0);
    }

    double DoubleCubicLogspace::operator()(const double x) const
    {
        return exp(inner_(x));
    }

    double DoubleCubicLogspace::derivative(const double x) const
    {
        return exp(inner_(x))*inner_.derivative(x);
    }

    double DoubleCubicLogspace::secondDerivative(const double x) const
    {
        const double deriv = inner_.derivative(x);
        return exp(inner_(x))*(inner_.secondDerivative(x) + deriv*deriv);
    }
}
