#ifndef ASE_GENERALISEDPOISSONHELPER_HH_
#define ASE_GENERALISEDPOISSONHELPER_HH_

#include "ase/AbsShiftableLogli.hh"

namespace ase {
    class GeneralisedPoisson;

    /** 
    // Helper class (positive skewness only) for the generalised
    // Poisson distribution
    */
    class GeneralisedPoissonHelper : public AbsShiftableLogli
    {
    public:
        GeneralisedPoissonHelper(double location, double sigPlus, double sigMinus);

        inline virtual ~GeneralisedPoissonHelper() override {}

        inline virtual GeneralisedPoissonHelper* clone() const override
            {return new GeneralisedPoissonHelper(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "GeneralisedPoissonHelper";}

    protected:
        inline virtual double unnormalizedMoment(
            const double p0, const unsigned n,
            const double maxDeltaLogli) const override
            {return smoothUnnormalizedMoment(p0, n, maxDeltaLogli);}

    private:
        friend class GeneralisedPoisson;

        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double s) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigmaBig_;
        double sigmaSmall_;
        double pmin_;
        double pmax_;
        double alpha_;
        double nu_;
        bool isSymmetric_;
    };
}

#endif // ASE_GENERALISEDPOISSONHELPER_HH_
