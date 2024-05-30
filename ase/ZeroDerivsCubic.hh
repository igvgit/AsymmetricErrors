#ifndef ASE_ZERODERIVSCUBIC_HH_
#define ASE_ZERODERIVSCUBIC_HH_

namespace ase {
    class ZeroDerivsCubic
    {
    public:
        // This cubic has value y0 at 0, value yx at x,
        // and both its first and second derivatives at x are 0.
        ZeroDerivsCubic(double y0, double x, double yx);

        double operator()(double x) const;
        double derivative(double x) const;
        double secondDerivative(double x) const;
        
    private:
        double a_;
        double b_;
        double c_;
        double d_;
    };
}

#endif // ASE_ZERODERIVSCUBIC_HH_
