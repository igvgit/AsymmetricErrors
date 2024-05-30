#include <sstream>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "ase/gCdfValues.hh"
#include "ase/AbsLocationScaleFamily.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/findRootUsingBisections.hh"

#define SQRTE 1.6487212707001281468

static double placeTheWall(
    const ase::AbsLocationScaleFamily& distro,
    const double tol, const bool moveRight)
{
    assert(tol > 0.0);
    const double wall = moveRight ? distro.quantile(1.0 - tol) :
                                    distro.quantile(tol);
    return wall;
}

namespace ase {
    double AbsLocationScaleFamily::unscaledMode() const
    {
        if (this->isDensityContinuous() && this->isUnimodal())
        {
            // Can try to use a generic algorithm here
            static const unsigned maxSteps = 1000;
            static const double tol = 2.0*std::numeric_limits<double>::epsilon();

            double xmin = this->unscaledQuantile(GCDF16);
            double xmax = this->unscaledQuantile(GCDF84);
            double dmin = this->unscaledDensityDerivative(xmin);
            double dmax = this->unscaledDensityDerivative(xmax);
            const double step = (xmax - xmin)/2.0;
            assert(step > 0.0);

            if (dmin*dmax >= 0.0)
            {
                assert(dmin*dmax > 0.0);
                const bool moveRight = dmax > 0.0;
                const double wall = placeTheWall(*this, tol, moveRight);

                for (unsigned i=0; i<maxSteps && dmin*dmax >= 0.0; ++i)
                {
                    if (moveRight)
                    {
                        xmin += step;
                        xmax += step;
                        if (xmax >= wall)
                            xmax = wall;
                        if (xmin >= wall)
                        {
                            if (!(this->unscaledDensity(wall) > 0.0 &&
                                  this->unscaledDensityDerivative(wall) > 0.0))
                                throw std::runtime_error(
                                    "In ase::AbsLocationScaleFamily::unscaledMode: "
                                    "density contitions are not satisfied at the positive wall. "
                                    "The wall is probably calculated incorrectly.");
                            return wall;
                        }
                    }
                    else
                    {
                        xmin -= step;
                        xmax -= step;
                        if (xmin <= wall)
                            xmin = wall;
                        if (xmax <= wall)
                        {
                            if (!(this->unscaledDensity(wall) > 0.0 &&
                                  this->unscaledDensityDerivative(wall) < 0.0))
                                throw std::runtime_error(
                                    "In ase::AbsLocationScaleFamily::unscaledMode: "
                                    "density contitions are not satisfied at the negative wall. "
                                    "The wall is probably calculated incorrectly.");
                            return wall;
                        }
                    }

                    dmin = this->unscaledDensityDerivative(xmin);
                    dmax = this->unscaledDensityDerivative(xmax);
                }
            }
            if (dmin*dmax >= 0.0) throw std::runtime_error(
                "In ase::AbsLocationScaleFamily::unscaledMode: "
                "failed to locate the mode");

            double mod = 0.0;
            const UnscaledDensityDerivativeFunctor1D f(*this);
            const bool status = findRootUsingBisections(
                f, 0.0, xmin, xmax, tol, &mod);
            assert(status);

            // Check that we are not looking at a dip for a bimodal density...
            const double h = 1.0e-5*step;
            double modp = mod + h;
            if (modp >= this->quantile(1.0))
                modp = mod;
            double modm = mod - h;
            if (modm <= this->quantile(0.0))
                modm = mod;
            assert(modp > modm);
            const double sder = (this->unscaledDensityDerivative(modp) -
                                 this->unscaledDensityDerivative(modm))/(modp - modm);
            if (sder >= 0.0) throw std::runtime_error(
                "In ase::AbsLocationScaleFamily::unscaledMode: "
                "second derivative is not negative");

            return mod;
        }
        else
        {
            std::ostringstream os;
            os << "In ase::" << this->classname()
               << "::unscaledMode: this function is not yet implemented";
            throw std::runtime_error(os.str());
            return 0.0;
        }
    }

    double AbsLocationScaleFamily::descentDelta(
        const bool isToTheRight, const double deltaLnL) const
    {
        if (!this->isUnimodal()) throw std::runtime_error(
            "In ase::AbsLocationScaleFamily::descentDelta: "
            "distribution is not unimodal");
        if (deltaLnL < 0.0) throw std::invalid_argument(
            "In ase::AbsLocationScaleFamily::descentDelta: "
            "deltaLnL argument must be non-negative");
        if (deltaLnL == 0.0)
            return 0.0;
        else
            return scale_*unscaledDescentDelta(isToTheRight, deltaLnL);
    }

    double AbsLocationScaleFamily::unscaledDescentDelta(
        const bool isToTheRight, const double deltaLnL) const
    {
        static const unsigned maxCycles = 1000;
        static const double tol = 2.0*std::numeric_limits<double>::epsilon();

        const double mode = this->unscaledMode();
        const double modeDensity = this->unscaledDensity(mode);
        assert(modeDensity > 0.0);
        const double descentFactor = deltaLnL == 0.5 ? SQRTE : exp(deltaLnL);
        const double targetDensity = modeDensity/descentFactor;
        assert(targetDensity < modeDensity);

        const double modeCdf = this->unscaledCdf(mode);
        if ((modeCdf == 0.0 && !isToTheRight) ||
            (modeCdf == 1.0 && isToTheRight))
            return 0.0;

        const int direction = isToTheRight ? 1 : -1;
        double cdfstep = isToTheRight ? (1.0 - modeCdf)/2.0 : modeCdf/2.0;
        double cdfCurrent = modeCdf;
        unsigned iCycle = 0;
        double x = 0.0;
        for (; iCycle<maxCycles; ++iCycle)
        {
            cdfCurrent += direction*cdfstep;
            cdfstep /= 2.0;
            x = this->unscaledQuantile(cdfCurrent);
            const double dens = this->unscaledDensity(x);
            if (dens == targetDensity)
                return std::abs(x - mode);
            if (dens < targetDensity)
                break;
        }
        assert(iCycle < maxCycles);

        const bool status = findRootUsingBisections(
            UnscaledDensityFunctor1D(*this), targetDensity, mode, x, tol, &x);
        assert(status);
        return std::abs(x - mode);
    }

    void AbsLocationScaleFamily::validateDeltas(
        const char* where, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        assert(where);
        if (deltaPlus <= 0.0 || deltaMinus <= 0.0)
        {
            std::ostringstream os;
            os.precision(17);
            os << "In " << where << ": descent deltas must be positive, "
               << "instead they were " << deltaPlus << " and " << deltaMinus;
            throw std::invalid_argument(os.str());
        }
        if (deltaLnL <= 0.0)
        {
            std::ostringstream os;
            os.precision(17);
            os << "In " << where << ": log likelihood delta must be positive, "
               << "instead it was " << deltaLnL;
            throw std::invalid_argument(os.str());
        }
    }

    double AbsLocationScaleFamily::qWidth() const
    {
        const double q16 = this->unscaledQuantile(GCDF16);
        const double q84 = this->unscaledQuantile(GCDF84);
        return scale_*(q84 - q16)/2.0;
    }

    double AbsLocationScaleFamily::qAsymmetry() const
    {
        const double q16 = this->unscaledQuantile(GCDF16);
        const double med = this->unscaledQuantile(0.5);
        const double q84 = this->unscaledQuantile(GCDF84);
        const double sp = q84 - med;
        const double sm = med - q16;
        return (sp - sm)/(sp + sm);
    }
}
