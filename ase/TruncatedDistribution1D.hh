#ifndef ASE_TRUNCATEDDISTRIBUTION1D_HH_
#define ASE_TRUNCATEDDISTRIBUTION1D_HH_

#include "ase/DistributionModel1DCopy.hh"

namespace ase {
    class TruncatedDistribution1D : public AbsDistributionModel1D
    {
    public:
        static const bool isFullOPAT = false;

        /**
        // Constructor arguments are as follows:
        //
        //  distro             -- distribution whose support we want to truncate
        //
        //  xmin, xmax         -- new limits for the support
        */
        TruncatedDistribution1D(const AbsDistributionModel1D& distro,
                                double xmin, double xmax);

        inline virtual TruncatedDistribution1D* clone() const override
            {return new TruncatedDistribution1D(*this);}

        inline virtual ~TruncatedDistribution1D() override {}

        virtual double density(double x) const override;
        virtual bool isDensityContinuous() const override;
        virtual bool isNonNegative() const override;
        virtual bool isUnimodal() const override;
        virtual double densityDerivative(double x) const override;
        virtual double cdf(double x) const override;
        virtual double exceedance(double x) const override;
        virtual double quantile(double x) const override;
        virtual double invExceedance(double x) const override;
        virtual double cumulant(unsigned n) const override;
        virtual double mode() const override;
        virtual double descentDelta(bool isToTheRight,
                                    double deltaLnL=0.5) const override;

        inline virtual std::string classname() const override
            {return "TruncatedDistribution1D";}

    private:
        DistributionModel1DCopy distro_;
        double xmin_;
        double xmax_;
        double cdfmin_;
        double cdfmax_;
        double exmin_;
    };
}

#endif // ASE_TRUNCATEDDISTRIBUTION1D_HH_
