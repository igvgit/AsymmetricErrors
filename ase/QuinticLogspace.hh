#ifndef ASE_QUINTICLOGSPACE_HH_
#define ASE_QUINTICLOGSPACE_HH_

#include "ase/QuinticInner.hh"

namespace ase {
    class QuinticLogspace
    {
    public:
        QuinticLogspace(double sigmaPlus, double yPlus,
                        double sigmaMinus, double yMinus);

        double operator()(double x) const;
        double derivative(double x) const;
        double secondDerivative(double x) const;

        inline const Poly1D& innerDerivPoly() const
            {return inner_.derivPoly();}

    private:
        QuinticInner inner_;
    };
}

#endif // ASE_QUINTICLOGSPACE_HH_
