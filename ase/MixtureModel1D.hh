#ifndef ASE_MIXTUREMODEL1D_HH_
#define ASE_MIXTUREMODEL1D_HH_

#include <vector>

#include "ase/DistributionModel1DCopy.hh"

namespace ase {
    class MixtureModel1D : public AbsDistributionModel1D
    {
    public:
        static const bool isFullOPAT = false;

        inline MixtureModel1D() : wsum_(0.0L), isNormalized_(false) {}

        inline virtual MixtureModel1D* clone() const override
            {return new MixtureModel1D(*this);}

        inline virtual ~MixtureModel1D() override {}

        /** 
        // Add a component to the mixture. Weight must be non-negative.
        // Addition of a component with zero weight is ignored.
        // All weights will be normalized internally so that their
        // sum is 1.
        */
        MixtureModel1D& add(const AbsDistributionModel1D& distro, double weight);

        inline unsigned nComponents() const {return entries_.size();}

        inline const AbsDistributionModel1D& getComponent(const unsigned n) const
            {return entries_.at(n).theCopy();}

        double getWeight(unsigned n) const;

        virtual double density(double x) const override;
        virtual bool isDensityContinuous() const override;
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
        virtual double random(AbsRNG& gen) const override;

        inline virtual std::string classname() const override
            {return "MixtureModel1D";}

    private:
        static const double tol_;
        static const double sqrtol_;

        void normalize();

        std::vector<DistributionModel1DCopy> entries_;
        std::vector<double> weights_;
        std::vector<double> weightCdf_;
        long double wsum_;
        bool isNormalized_;
    };
}

#endif // ASE_MIXTUREMODEL1D_HH_
