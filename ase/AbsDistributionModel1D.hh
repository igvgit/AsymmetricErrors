#ifndef ASE_ABSDISTRIBUTIONMODEL1D_HH_
#define ASE_ABSDISTRIBUTIONMODEL1D_HH_

#include <string>

#include "ase/AbsRNG.hh"

namespace ase {
    /** Base class for univariate distribution models */
    struct AbsDistributionModel1D
    {
        // If you add any new functions here, add the corresponding
        // forwarding call to the class DistributionModel1DCopy
        inline virtual ~AbsDistributionModel1D() {}

        /** "Virtual copy constructor" */
        virtual AbsDistributionModel1D* clone() const = 0;

        /** Probability density */
        virtual double density(double x) const = 0;

        /**
        // Distributions whose density is not continuous
        // should override the following
        */
        inline virtual bool isDensityContinuous() const {return true;}

        /**
        // Distributions whose density becomes negative
        // somewhere should override the following
        */
        inline virtual bool isNonNegative() const {return true;}

        /**
        // Distributions that are not unimodal should
        // override the following
        */
        inline virtual bool isUnimodal() const {return true;}

        /** Probability density derivative */
        virtual double densityDerivative(double x) const = 0;

        /** Cumulative distribution function */
        virtual double cdf(double x) const = 0;

        /**
        // 1 - cdf, known as "survival function" or "exceedance".
        // Implementations should avoid subtractive cancellation.
        */
        virtual double exceedance(double x) const = 0;

        /**
        // The quantile function (inverse cdf). quantile(0.0)
        // and quantile(1.0) should return the effective support
        // boundaries, taking into account the numerical precision
        // of the calculations.
        */
        virtual double quantile(double x) const = 0;

        /**
        // quantile(1 - x), or inverse exceedance function.
        // Implementations should avoid subtractive cancellation.
        */
        virtual double invExceedance(double x) const = 0;

        /**
        // Distribution cumulants. It is expected that the derived
        // models should implement cumulant calculations at least
        // up to and including n = 4.
        */
        virtual double cumulant(unsigned n) const = 0;

        /** Location of the distribution mode */
        virtual double mode() const = 0;

        /** Distance to the mode in x for the given delta loglikelihood */
        virtual double descentDelta(bool isToTheRight,
                                    double deltaLnL=0.5) const = 0;

        /** Class name for various printouts */
        virtual std::string classname() const = 0;

        /**
        // Generate random numbers according to this distribution.
        // This function is virtual because often there are better
        // ways to generate random numbers from a given distribution
        // than to call its quantile or invExceedance methods.
        */
        virtual double random(AbsRNG& gen) const;

        /** Quantile-based width */
        virtual double qWidth() const;
        
        /** Quantile-based asymmetry */
        virtual double qAsymmetry() const;
    };
}

#endif //ASE_ABSDISTRIBUTIONMODEL1D_HH_
