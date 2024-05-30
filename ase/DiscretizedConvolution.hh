#ifndef ASE_DISCRETIZEDCONVOLUTION_HH_
#define ASE_DISCRETIZEDCONVOLUTION_HH_

#include <vector>

#include "ase/AbsDistributionModel1D.hh"
#include "ase/TabulatedDensity1D.hh"
#include "ase/InterpolatedDensity1D.hh"

namespace ase {
    /**
    // Numerical convolution of two densities obtained by discretizing
    // the support interval
    */
    class DiscretizedConvolution
    {
    public:
        /**
        // xmin and xmax arguments are used to model the suppport of
        // density m2 and of the result. Density m1 will be discretized
        // using the same step size but on a different interval. The
        // discretization will be performed by calculating the average
        // density with calls to the "cdf" method. The scan of the
        // densities requires O(nIntervals) operations while subsequent
        // construction of the convolution is performed brute-force,
        // in O(nIntervals^2) steps. Once the object is constructed,
        // operator() is very fast, O(1).
        //
        // Not normalizing at the beginning can be helpful for estimating
        // the quality of the convolution (i.e., the fraction of the
        // distribution lost because the interval [xmin, xmax] did not
        // cover the support of m2 or of the result).
        */
        DiscretizedConvolution(const AbsDistributionModel1D& m1,
                               const AbsDistributionModel1D& m2,
                               double xmin, double xmax,
                               unsigned nIntervals, bool normalize = false);

        /** Normalize the object as a probability density */
        void normalize();

        /**
        // The density returned by this operator will be piecewise constant.
        // If you want smooth interpolation (which, of course, will be
        // approximate), call one of the "construct density" methods.
        */
        double operator()(double x) const;

        /**
        // Density integral will be very close to 1 if the distribution
        // is normalized
        */
        double densityIntegral() const;

        inline double xmin() const {return xmin_;}
        inline double xmax() const {return xmax_;}
        inline unsigned nIntervals() const {return nIntervals_;}
        inline double intervalWidth() const
            {return (xmax_ - xmin_)/nIntervals_;}
        inline bool isNormalized() const {return normalized_;}
        inline double convolvedValue(const unsigned i) const
            {return convolution_.at(i);}

        /** The following method returns the interval center */
        double coordinateAt(unsigned i) const;

        // The "construct density" methods will automatically choose
        // an appropriate grid
        TabulatedDensity1D constructTabulatedDensity() const;
        InterpolatedDensity1D constructInterpolatedDensity() const;

    private:
        std::vector<double> convolution_;
        double xmin_;
        double xmax_;
        unsigned nIntervals_;
        bool normalized_;
    };
}

#endif // ASE_DISCRETIZEDCONVOLUTION_HH_
