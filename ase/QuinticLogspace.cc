#include <cmath>
#include <cassert>

#include "ase/QuinticLogspace.hh"

namespace ase {
    QuinticLogspace::QuinticLogspace(
        const double sigmaPlus, const double yPlus,
        const double sigmaMinus, const double yMinus)
        : inner_(sigmaPlus, log(yPlus), sigmaMinus, log(yMinus))
    {
        assert(yPlus > 0.0);
        assert(yMinus > 0.0);
    }

    double QuinticLogspace::operator()(const double x) const
    {
        return exp(inner_(x));
    }

    double QuinticLogspace::derivative(const double x) const
    {
        return exp(inner_(x))*inner_.derivative(x);
    }

    double QuinticLogspace::secondDerivative(const double x) const
    {
        const double deriv = inner_.derivative(x);
        return exp(inner_(x))*(inner_.secondDerivative(x) + deriv*deriv);
    }
}
