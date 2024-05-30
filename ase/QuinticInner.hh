#ifndef ASE_QUINTICINNER_HH_
#define ASE_QUINTICINNER_HH_

#include "ase/Poly1D.hh"

namespace ase {
    class QuinticInner
    {
    public:
        // This quintic polynomial has value yPlus at sigmaPlus
        // and value yMinus at -sigmaMinus. Both first and second
        // derivative at both sigma points are 0.
        // The function is continued by constants beyond the
        // range [-sigmaMinus, sigmaPlus]. These constants
        // are, naturally, yPlus for x >= sigmaPlus_ and yMinus
        // for x <= -sigmaMinus. The function itself is continuous
        // everywhere, together with its first and second derivatives.
        // The third derivative is normally discontinuous at
        // x = -sigmaMinus and x = sigmaPlus.
        QuinticInner(double sigmaPlus, double yPlus,
                     double sigmaMinus, double yMinus);

        double operator()(double x) const;
        double derivative(double x) const;
        double secondDerivative(double x) const;

        inline const Poly1D& derivPoly() const
            {return deriv_;}

    private:
        double sigmaPlus_;
        double yPlus_;
        double sigmaMinus_;
        double yMinus_;
        Poly1D quintic_;
        Poly1D deriv_;
        Poly1D sder_;
    };
}

#endif // ASE_QUINTICINNER_HH_
