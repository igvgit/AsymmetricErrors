#include <stdexcept>

#include "ase/ZeroDerivsCubic.hh"

namespace ase {
    ZeroDerivsCubic::ZeroDerivsCubic(
        const double y0, const double x, const double yx)
        : d_(y0)
    {
        if (!x) throw std::invalid_argument(
            "In ase::ZeroDerivsCubic constructor: "
            "argument x must not be 0");
        const double del = y0 - yx;
        const double x2 = x*x;
        a_ = -del/x2/x;
        b_ = 3.0*del/x2;
        c_ = -3.0*del/x;
    }

    double ZeroDerivsCubic::operator()(const double x) const
    {
        return ((a_*x + b_)*x + c_)*x + d_;
    }

    double ZeroDerivsCubic::derivative(const double x) const
    {
        return (3.0*a_*x + 2.0*b_)*x + c_;
    }

    double ZeroDerivsCubic::secondDerivative(const double x) const
    {
        return 2.0*b_ + 6.0*a_*x;
    }
}
