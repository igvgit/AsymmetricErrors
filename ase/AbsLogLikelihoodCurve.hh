#ifndef ASE_ABSLOGLIKELIHOODCURVE_HH_
#define ASE_ABSLOGLIKELIHOODCURVE_HH_

#include <string>
#include <cassert>
#include <utility>

namespace ase {
    /** Base class for log-likelihood curves */
    class AbsLogLikelihoodCurve
    {
    public:
        // If you add any new functions here, add the corresponding
        // forwarding call to the class LikelihoodCurveCopy
        inline virtual ~AbsLogLikelihoodCurve() {}

        /** "Virtual copy constructor" */
        virtual AbsLogLikelihoodCurve* clone() const = 0;

        /** Left boundary of the support interval */
        virtual double parMin() const = 0;

        /** Right boundary of the support interval */
        virtual double parMax() const = 0;

        /**
        // Typical "location" parameter of the curve. Should normally
        // coincide with "argmax()" for concave curves.
        */
        virtual double location() const = 0;

        /**
        // A length scale in the parameter space on which the curve
        // shape does not deviate appreciably from a straight line.
        // For curves that involve interpolation, this should be the
        // distance between the nodes. It is also expected that
        // numerical calculation of second derivative at the maximum
        // (located not at the edge of the support) with a fraction
        // of this step size (1/10th or so) should return a negative
        // number.
        */
        virtual double stepSize() const = 0;

        /** Maximum likelihood value */
        virtual double maximum() const = 0;

        /** Parameter value that corresponds to likelihood maximum */
        virtual double argmax() const = 0;

        /** Likelihood value */
        virtual double operator()(double parameter) const = 0;

        /** Likelihood derivative */
        virtual double derivative(double parameter) const = 0;

        /**
        // Likelihood second derivative. May or may not be numeric.
        // Default step value of 0 means that the step size for
        // numerical differentiation will be proportional to the
        // value returned by the "stepSize()" method.
        */
        virtual double secondDerivative(double param, double step = 0.0) const;

        /** Class name for various printouts */
        virtual std::string classname() const = 0;

        /** In-place multiplication by a constant */
        virtual AbsLogLikelihoodCurve& operator*=(double c) = 0;
        inline virtual AbsLogLikelihoodCurve& operator/=(const double c)
            {assert(c); return *this *= (1.0/c);}

        // stepFactor should be 1.0 or larger. The code will make
        // steps away from the maximum to find a point at which
        // the likelihood decreases by more than deltaLogLikelihood.
        // Each time the step will be increased by this factor.
        virtual double sigmaPlus(double deltaLogLikelihood = 0.5,
                                 double stepFactor = 1.1) const;

        virtual double sigmaMinus(double deltaLogLikelihood = 0.5,
                                  double stepFactor = 1.1) const;

        // The following function returns the location of the
        // maximum as the first element of the pair and its
        // value as the second. If a local maximum is not found
        // (for example, if the curve is convex), the second
        // element will be set to -DBL_MAX.
        //
        // Applications should typically use the "maximum"
        // function instead of this one. Only if the "maximum"
        // fails to find the global maximum, it would make
        // sense to try to search for it in multiple places
        // using this method.
        virtual std::pair<double,double> findLocalMaximum(
            double startingPoint, bool searchToTheRight,
            unsigned maxSteps, double stepFactor = 1.1) const;

        // Posterior mean and variance using a flat prior for the parameter
        virtual double posteriorMean() const;
        virtual double posteriorVariance() const;

        // In-place additions and subtractions are not declared here
        // because they can involve redefinition of the support.
        //
        // Providing I/O would probably be an overkill for this package.
    protected:
        // Calculate expectation value of (p - p0)^n using
        // a flat prior for the parameter
        virtual double posteriorMoment(double p0, unsigned n) const;

        virtual double unnormalizedMoment(double p0, unsigned n,
                                          double maxDeltaLogli) const;
    };

    /** Functor for AbsLogLikelihoodCurve derivative */
    class LogLikelihoodDerivative
    {
    public:
        inline explicit LogLikelihoodDerivative(const AbsLogLikelihoodCurve& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.derivative(x);}

    private:
        const AbsLogLikelihoodCurve& fcn_;
    };

    /** Functor for AbsLogLikelihoodCurve second derivative */
    class LogLikelihoodSecondDerivative
    {
    public:
        inline explicit LogLikelihoodSecondDerivative(
            const AbsLogLikelihoodCurve& fcn, const double step)
            : fcn_(fcn), step_(step) {}

        inline double operator()(const double x) const
            {return fcn_.secondDerivative(x, step_);}

    private:
        const AbsLogLikelihoodCurve& fcn_;
        double step_;
    };
}

#endif // ASE_ABSLOGLIKELIHOODCURVE_HH_
