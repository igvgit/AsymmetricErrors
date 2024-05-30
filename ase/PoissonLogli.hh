#ifndef ASE_POISSONLOGLI_HH_
#define ASE_POISSONLOGLI_HH_

#include <cfloat>
#include <algorithm>

#include "ase/AbsLogLikelihoodCurve.hh"

namespace ase {
    class PoissonLogli : public AbsLogLikelihoodCurve
    {
    public:
        inline PoissonLogli(const unsigned long i_n)
            : n_(i_n), dn_(i_n), factor_(1.0) {}

        inline virtual ~PoissonLogli() override {}

        inline virtual PoissonLogli* clone() const override
            {return new PoissonLogli(*this);}

        inline unsigned long n() const
            {return n_;}

        inline virtual double parMin() const override
            {return 0.0;}

        inline virtual double parMax() const override
            {return DBL_MAX;}

        inline virtual double location() const override
            {return dn_;}

        inline virtual double stepSize() const override
            {return 0.1*std::max(sqrt(dn_), 1.0);}

        inline virtual double maximum() const override
            {return 0.0;}

        inline virtual double argmax() const override
            {return dn_;}

        virtual double operator()(double lam) const override;
        virtual double derivative(double lam) const override;
        virtual double secondDerivative(double lam, double step = 0.0) const override;

        inline virtual std::string classname() const
            {return "PoissonLogli";}

        inline virtual AbsLogLikelihoodCurve& operator*=(const double c) override
            {factor_ *= c; return *this;}

        inline virtual double sigmaMinus(double deltaLogLi = 0.5,
                                         double stepFactor = 1.1) const override
        {
            if (n_)
                return AbsLogLikelihoodCurve::sigmaMinus(deltaLogLi, stepFactor);
            else
                return 0.0;
        }

        inline virtual double posteriorMean() const override
            {return dn_ + 1.0;}

        inline virtual double posteriorVariance() const override
            {return dn_ + 1.0;}

    private:
        unsigned long n_;
        double dn_;
        double factor_;
    };
}

#endif // ASE_POISSONLOGLI_HH_
