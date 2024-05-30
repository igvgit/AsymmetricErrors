#ifndef ASE_ABSLOCATIONSCALEFAMILY_HH_
#define ASE_ABSLOCATIONSCALEFAMILY_HH_

#include <cmath>
#include <stdexcept>

#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    class UnscaledDensityFunctor1D;
    class UnscaledDensityDerivativeFunctor1D;
    class UnscaledCdfFunctor1D;
    class UnscaledExceedanceFunctor1D;
    class UnscaledQuantileFunctor1D;
    class UnscaledInvExceedanceFunctor1D;
    class UnscaledMomentFunctor1D;
    class UnscaledEntropyFunctor1D;

    /**
    // Base class for univariate distribution models
    // that can be shifted and scaled
    */
    class AbsLocationScaleFamily : public AbsDistributionModel1D
    {
    public:
        /** Location and scale parameters must be provided */
        inline AbsLocationScaleFamily(const double location,
                                      const double scale)
            : location_(location), scale_(scale)
        {
            if (scale_ <= 0.0) throw std::invalid_argument(
                "In ase::AbsLocationScaleFamily constructor: "
                "scale parameter must be positive");
        }

        /** "Virtual copy constructor" */
        virtual AbsLocationScaleFamily* clone() const = 0;

        inline virtual ~AbsLocationScaleFamily() override {}

        /** Get the location parameter */
        inline double location() const {return location_;}

        /** Get the scale parameter */
        inline double scale() const {return scale_;}

        /** Set the location parameter */
        inline virtual void setLocation(const double loc) {location_ = loc;}

        /** Set the scale parameter */
        inline virtual void setScale(const double s)
        {
            if (s <= 0.0) throw std::invalid_argument(
                "In ase::AbsLocationScaleFamily::setScale: "
                "scale parameter must be positive");
            scale_ = s;
        }

        //@{
        /** Method overriden from the AbsDistributionModel1D base class */
        inline double density(const double x) const override
            {return unscaledDensity((x - location_)/scale_)/scale_;}

        inline double densityDerivative(const double x) const override
            {return unscaledDensityDerivative((x - location_)/scale_)/scale_/scale_;}

        inline double cdf(const double x) const override
            {return unscaledCdf((x - location_)/scale_);}

        inline double exceedance(const double x) const override
            {return unscaledExceedance((x - location_)/scale_);}

        inline double quantile(const double x) const override
            {return scale_*unscaledQuantile(x) + location_;}

        inline double invExceedance(const double x) const override
            {return scale_*unscaledInvExceedance(x) + location_;}

        inline double cumulant(const unsigned n) const override
        {
            switch (n)
            {
            case 0U:
                return 1.0;
            case 1U:
                return scale_*unscaledCumulant(1U) + location_;
            case 2U:
                return scale_*scale_*unscaledCumulant(2U);
            default:
                return pow(scale_, n)*unscaledCumulant(n);
            }
        }

        inline double mode() const override
            {return scale_*unscaledMode() + location_;}

        double descentDelta(bool isToTheRight,
                            double deltaLnL=0.5) const override;

        inline double random(AbsRNG& gen) const override
            {return unscaledRandom(gen)*scale_ + location_;}
        //@}

        /** Distribution width using 16th and 84th percentiles */
        double qWidth() const override;

        double qAsymmetry() const override;

        virtual std::string classname() const = 0;

    protected:
        static void validateDeltas(const char* where, double deltaPlus,
                                   double deltaMinus, double deltaLnL);
    private:
        friend class UnscaledDensityFunctor1D;
        friend class UnscaledDensityDerivativeFunctor1D;
        friend class UnscaledCdfFunctor1D;
        friend class UnscaledExceedanceFunctor1D;
        friend class UnscaledQuantileFunctor1D;
        friend class UnscaledInvExceedanceFunctor1D;
        friend class UnscaledMomentFunctor1D;
        friend class UnscaledEntropyFunctor1D;

        virtual double unscaledDensity(double x) const = 0;
        virtual double unscaledDensityDerivative(double x) const = 0;
        virtual double unscaledCdf(double x) const = 0;
        virtual double unscaledExceedance(double x) const = 0;
        virtual double unscaledQuantile(double x) const = 0;
        virtual double unscaledInvExceedance(const double x) const = 0;
        virtual double unscaledCumulant(unsigned n) const = 0;
        virtual double unscaledMode() const;
        virtual double unscaledDescentDelta(bool isToTheRight,
                                            double deltaLnL) const;
        inline virtual double unscaledRandom(AbsRNG& gen) const
            {return unscaledQuantile(gen());}

        double location_;
        double scale_;
    };
}

#endif // ASE_ABSLOCATIONSCALEFAMILY_HH_
