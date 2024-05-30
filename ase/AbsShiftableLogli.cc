#include <cfloat>
#include <stdexcept>
#include <sstream>

#include "ase/AbsShiftableLogli.hh"
#include "ase/PosteriorMomentFunctor.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace ase {
    AbsShiftableLogli::AbsShiftableLogli(const double i_shift)
        : shift_(i_shift), factor_(1.0)
    {
    }

    double AbsShiftableLogli::parMin() const
    {
        const double lim = uParMin();
        if (lim == -DBL_MAX)
            return lim;
        else
            return lim + shift_;
    }

    double AbsShiftableLogli::parMax() const
    {
        const double lim = uParMax();
        if (lim == DBL_MAX)
            return lim;
        else
            return lim + shift_;
    }

    double AbsShiftableLogli::maximum() const
    {
        if (factor_ <= 0.0) throw std::runtime_error(
            "In ase::AbsShiftableLogli::maximum: "
            "second derivative is not negative");
        return factor_*uMaximum();
    }

    double AbsShiftableLogli::argmax() const
    {
        if (factor_ <= 0.0) throw std::runtime_error(
            "In ase::AbsShiftableLogli::argmax: "
            "second derivative is not negative");
        return uArgmax() + shift_;
    }

    double AbsShiftableLogli::sigmaPlus(const double deltaLogLikelihood,
                                        const double f) const
    {
        if (factor_ <= 0.0) throw std::runtime_error(
            "In ase::AbsShiftableLogli::sigmaPlus: "
            "second derivative is not negative");
        if (deltaLogLikelihood < 0.0) throw std::invalid_argument(
            "In ase::AbsShiftableLogli::sigmaPlus: "
            "deltaLogLikelihood argument must be non-negative");
        if (deltaLogLikelihood == 0.0)
            return 0.0;
        else
            return uSigmaPlus(deltaLogLikelihood/factor_, f);
    }

    double AbsShiftableLogli::sigmaMinus(const double deltaLogLikelihood,
                                         const double f) const
    {
        if (factor_ <= 0.0) throw std::runtime_error(
            "In ase::AbsShiftableLogli::sigmaMinus: "
            "second derivative is not negative");
        if (deltaLogLikelihood < 0.0) throw std::invalid_argument(
            "In ase::AbsShiftableLogli::sigmaMinus: "
            "deltaLogLikelihood argument must be non-negative");
        if (deltaLogLikelihood == 0.0)
            return 0.0;
        else
            return uSigmaMinus(deltaLogLikelihood/factor_, f);
    }

    void AbsShiftableLogli::validateSigmas(
        const char* where, const double sigmaPlus, const double sigmaMinus)
    {
        assert(where);
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0)
        {
            std::ostringstream os;
            os.precision(17);
            os << "In " << where << ": sigma parameters must be positive, "
               << "instead they were " << sigmaPlus << " and " << sigmaMinus;
            throw std::invalid_argument(os.str());
        }
    }

    double AbsShiftableLogli::uSecondDerivative(
        const double p, double i_step) const
    {
        if (!i_step)
            i_step = 0.1*stepSize();
        const double pmin = uParMin();
        const double pmax = uParMax();
        assert(pmin < pmax);
        if (p < pmin || p > pmax) throw std::invalid_argument(
            "In ase::AbsShiftableLogli::uSecondDerivative: "
            "argument out of range");
        const double step = std::abs(i_step);
        assert(step);
        volatile double pplus = p + step;
        if (pplus > pmax)
            pplus = pmax;
        volatile double pminus = p - step;
        if (pminus < pmin)
            pminus = pmin;
        return (uDerivative(pplus) - uDerivative(pminus))/(pplus - pminus);
    }

    double AbsShiftableLogli::smoothUnnormalizedMoment(
        const double p0, const unsigned n, const double maxDeltaLogli) const
    {
        double pmin, pmax;
        const double maxArg = uArgmax();
        try {
            pmin = maxArg - uSigmaMinus(maxDeltaLogli, 1.1);
        }
        catch (const std::invalid_argument&) {
            pmin = uParMin();
            assert(pmin > -DBL_MAX);
        }
        try {
            pmax = maxArg + uSigmaPlus(maxDeltaLogli, 1.1);
        }
        catch (const std::invalid_argument&) {
            pmax = uParMax();
            assert(pmax < DBL_MAX);
        }
        assert(pmin < pmax);
        const unsigned nIntervals = (pmax - pmin)/stepSize() + 1.0;
        const GaussLegendreQuadrature glq(4U);
        const PosteriorMomentFunctor momFcn(*this, p0, n);
        const double s = shift();
        return glq.integrate(momFcn, pmin+s, pmax+s, nIntervals);
    }
}
