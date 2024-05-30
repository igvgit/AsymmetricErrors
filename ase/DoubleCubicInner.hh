#ifndef ASE_DOUBLECUBICINNER_HH_
#define ASE_DOUBLECUBICINNER_HH_

#include "ase/ZeroDerivsCubic.hh"

namespace ase {
    class DoubleCubicInner
    {
    public:
        // This double cubic has value y0 at 0, value yPlus
        // at sigmaPlus, value yMinus at -sigmaMinus. Both
        // first and second derivatives at both sigmas are 0.
        // The function is continued by constants beyond the
        // range [-sigmaMinus, sigmaPlus]. These constants
        // are, naturally, yPlus for x >= sigmaPlus_ and yMinus
        // for x <= -sigmaMinus. The function itself is continuous
        // everywhere. Its first and second derivatives are
        // continuous everywhere except at x = 0 where the first
        // derivative is normally discontinuous.
        DoubleCubicInner(double y0, double sigmaPlus, double yPlus,
                         double sigmaMinus, double yMinus);

        double operator()(double x) const;
        double derivative(double x) const;
        double secondDerivative(double x) const;

    private:
        double sigmaPlus_;
        double yPlus_;
        double sigmaMinus_;
        double yMinus_;
        ZeroDerivsCubic left_;
        ZeroDerivsCubic right_;
    };
}

#endif // ASE_DOUBLECUBICINNER_HH_
