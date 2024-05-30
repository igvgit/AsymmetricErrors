#include <cmath>
#include <cfloat>
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "ase/findRootUsingBisections.hh"
#include "ase/PosteriorMomentFunctor.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace {
    double findSigma(const ase::AbsLogLikelihoodCurve& curve,
                     const double deltaLogLikelihood,
                     const double stepFactor,
                     const int direction)
    {
        static const double eps = 2.0*std::numeric_limits<double>::epsilon();

        if (deltaLogLikelihood < 0.0) throw std::invalid_argument(
            "In findSigma: deltaLogLikelihood argument must be non-negative");
        if (deltaLogLikelihood == 0.0)
            return 0.0;
        if (stepFactor < 1.0) throw std::invalid_argument(
            "In findSigma: step factor must be at least 1.0");
        const double maxArg = curve.argmax();
        const double xmin = curve.parMin();
        const double xmax = curve.parMax();
        const double target = curve.maximum() - deltaLogLikelihood;
        double step = curve.stepSize();
        assert(step > 0.0);
        double xabove = maxArg;
        double xbelow = 0.0;
        bool boundaryReached = false, xbelowFound = false;
        while (!(boundaryReached || xbelowFound))
        {
            xbelow = xabove + step*direction;
            if (direction > 0)
            {
                if (xbelow >= xmax)
                {
                    xbelow = xmax;
                    boundaryReached = true;
                }
            }
            else
            {
                if (xbelow <= xmin)
                {
                    xbelow = xmin;
                    boundaryReached = true;
                }
            }
            const double logli = curve(xbelow);
            if (logli == target)
                return std::abs(xbelow - maxArg);
            if (logli < target)
                xbelowFound = true;
            else
                xabove = xbelow;
            step *= stepFactor;
        }

        if (!xbelowFound)
        {
            assert(boundaryReached);
            throw std::invalid_argument(
                "In findSigma: deltaLogLikelihood argument is too large");
        }

        if (xbelow > xabove)
            std::swap(xbelow, xabove);

        double xSigma;
        const bool status = ase::findRootUsingBisections(
            curve, target, xbelow, xabove, eps, &xSigma);
        assert(status);
        return std::abs(xSigma - maxArg);
    }
}

namespace ase {
    double AbsLogLikelihoodCurve::sigmaPlus(const double deltaLogLikelihood,
                                            const double stepFactor) const
    {
        return findSigma(*this, deltaLogLikelihood, stepFactor, 1);
    }

    double AbsLogLikelihoodCurve::sigmaMinus(const double deltaLogLikelihood,
                                             const double stepFactor) const
    {
        return findSigma(*this, deltaLogLikelihood, stepFactor, -1);
    }

    std::pair<double,double> AbsLogLikelihoodCurve::findLocalMaximum(
        const double start, const bool moveRight,
        const unsigned maxSteps, const double stepFactor) const
    {
        static const double eps = 2.0*DBL_EPSILON;

        const double direction = moveRight ? 1.0 : -1.0;
        const double pmin = parMin();
        const double pmax = parMax();
        assert(pmin < pmax);
        const double step0 = stepSize();
        double step = step0;
        assert(step > 0.0);
        double xold = start;
        double dold = derivative(xold);
        if (dold == 0.0)
        {
            if (secondDerivative(xold) < 0.0)
                return std::pair<double,double>(xold, (*this)(xold));
            else
                return std::pair<double,double>(xold, -DBL_MAX);
        }
        double xnew = 0.0;
        bool boundaryReached = false, intervalFound = false;
        for (unsigned istep=0;
             istep<maxSteps && (!(boundaryReached || intervalFound));
             ++istep)
        {
            xnew = xold + step*direction;
            if (direction > 0)
            {
                if (xnew >= pmax)
                {
                    xnew = pmax;
                    boundaryReached = true;
                }
            }
            else
            {
                if (xnew <= pmin)
                {
                    xnew = pmin;
                    boundaryReached = true;
                }
            }
            const double dnew = derivative(xnew);
            if (dnew == 0.0)
            {
                if (secondDerivative(xnew) < 0.0)
                    return std::pair<double,double>(xnew, (*this)(xnew));
                else
                    return std::pair<double,double>(xnew, -DBL_MAX);
            }
            if (dold*dnew < 0.0)
            {
                // The derivative switched sign
                intervalFound = true;
            }
            else
            {
                xold = xnew;
                dold = dnew;
            }
            step *= stepFactor;
        }

        if (!intervalFound)
            return std::pair<double,double>(xnew, -DBL_MAX);

        if (xold > xnew)
            std::swap(xold, xnew);

        double extremum;
        const bool status = ase::findRootUsingBisections(
            LogLikelihoodDerivative(*this), 0.0, xold, xnew, eps, &extremum);
        assert(status);
            
        if (secondDerivative(extremum) < 0.0)
            return std::pair<double,double>(extremum, (*this)(extremum));
        else
            return std::pair<double,double>(extremum, -DBL_MAX);
    }

    double AbsLogLikelihoodCurve::secondDerivative(const double p,
                                                   double i_step) const
    {
        if (!i_step)
            i_step = 0.1*stepSize();
        const double pmin = parMin();
        const double pmax = parMax();
        assert(pmin < pmax);
        if (p < pmin || p > pmax) throw std::invalid_argument(
            "In ase::AbsLogLikelihoodCurve::secondDerivative: "
            "argument out of range");
        const double step = std::abs(i_step);
        assert(step);
        volatile double pplus = p + step;
        if (pplus > pmax)
            pplus = pmax;
        volatile double pminus = p - step;
        if (pminus < pmin)
            pminus = pmin;
        return (derivative(pplus) - derivative(pminus))/(pplus - pminus);
    }

    double AbsLogLikelihoodCurve::unnormalizedMoment(
        const double p0, const unsigned n, const double maxDeltaLogli) const
    {
        // Break the integral into four different pieces because
        // many models have discontinuous derivatives of various
        // orders at 0, sigmaPlus, and sigmaMinus.
        double boundaries[5];
        const double maxArg = argmax();
        boundaries[0] = maxArg - sigmaMinus(maxDeltaLogli);
        boundaries[1] = maxArg - sigmaMinus();
        boundaries[2] = maxArg;
        boundaries[3] = maxArg + sigmaPlus();
        boundaries[4] = maxArg + sigmaPlus(maxDeltaLogli);
        const GaussLegendreQuadrature glq(4U);
        const PosteriorMomentFunctor momFcn(*this, p0, n);
        const double h = stepSize();
        assert(h > 0.0);
        long double sum = 0.0L;
        for (unsigned i=0; i<4U; ++i)
        {
            assert(boundaries[i+1] >= boundaries[i]);
            const unsigned nIntervals = (boundaries[i+1] - boundaries[i])/h + 1.0;
            sum += glq.integrate(momFcn, boundaries[i], boundaries[i+1], nIntervals);
        }
        return sum;
    }

    double AbsLogLikelihoodCurve::posteriorMoment(
        const double p0, const unsigned n) const
    {
        static const double nIntegSigmas = 10.0;

        if (n)
        {
            const double maxDeltaLogli = nIntegSigmas*nIntegSigmas/2.0;
            const double i0 = unnormalizedMoment(p0, 0U, maxDeltaLogli);
            assert(i0 > 0.0);
            const double in = unnormalizedMoment(p0, n, maxDeltaLogli);
            return in/i0;
        }
        else
            return 1.0;
    }

    double AbsLogLikelihoodCurve::posteriorMean() const
    {
        const double maxArg = argmax();
        return maxArg + posteriorMoment(maxArg, 1U);
    }

    double AbsLogLikelihoodCurve::posteriorVariance() const
    {
        const double mu = posteriorMean();
        return posteriorMoment(mu, 2U);
    }
}
