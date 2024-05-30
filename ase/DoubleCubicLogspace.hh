#ifndef ASE_DOUBLECUBICLOGSPACE_HH_
#define ASE_DOUBLECUBICLOGSPACE_HH_

#include "ase/DoubleCubicInner.hh"

namespace ase {
    class DoubleCubicLogspace
    {
    public:
        DoubleCubicLogspace(double y0, double sigmaPlus, double yPlus,
                            double sigmaMinus, double yMinus);

        double operator()(double x) const;
        double derivative(double x) const;
        double secondDerivative(double x) const;

    private:
        DoubleCubicInner inner_;
    };
}

#endif // ASE_DOUBLECUBICLOGSPACE_HH_
