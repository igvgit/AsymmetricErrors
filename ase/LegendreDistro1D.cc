#include <cfloat>
#include <stdexcept>

#include "ase/LegendreDistro1D.hh"
#include "ase/Poly1D.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/DistributionFunctors1D.hh"

namespace ase {
    bool LegendreDistro1D::isUnimodal() const
    {
        throw std::runtime_error("In ase::LegendreDistro1D::isUnimodal: "
                                 "this method is not implemented");
        return false;
    }

    void LegendreDistro1D::setupCoeffs(const double* coeffs,
                                       const unsigned maxdeg)
    {
        if (maxdeg)
            assert(coeffs);
        allCoeffs_.resize(maxdeg + 1U);
        allCoeffs_[0] = 0.5;
        for (unsigned i=0; i<maxdeg; ++i)
            allCoeffs_[i+1U] = coeffs[i];
        for (unsigned i=maxdeg; i; --i)
        {
            if (allCoeffs_[i] == 0.0)
                allCoeffs_.pop_back();
            else
                break;
        }

        const unsigned deg = allCoeffs_.size() - 1;
        integCoeffs_.resize(deg + 2U);
        poly_.integralCoeffs(&allCoeffs_[0], deg, &integCoeffs_[0]);
        if (deg)
        {
            derivCoeffs_.resize(deg);
            poly_.derivativeCoeffs(&allCoeffs_[0], deg, &derivCoeffs_[0]);
        }
    }

    void LegendreDistro1D::checkPositive(const bool validatePositivity)
    {
        const unsigned sz = allCoeffs_.size();
        std::vector<long double> monoCoeffs(sz);
        poly_.monomialCoeffs(&allCoeffs_[0], sz-1U, &monoCoeffs[0]);
        Poly1D monoPoly(monoCoeffs);
        if (monoPoly.nRoots(-1.0L, 1.0L))
        {
            isPositive_ = false;
            if (validatePositivity) throw std::invalid_argument(
                "In ase::LegendreDistro1D::checkPositive: "
                "the density is not strictly positive on [-1, 1]");
        }
        else
            isPositive_ = true;
    }

    void LegendreDistro1D::calculateCumulants() const
    {
        cumulants_[0] = 1.0;
        const double mu = calculateMoment(0.0, 1U);
        cumulants_[1] = mu;
        const double var = calculateMoment(mu, 2U);
        cumulants_[2] = var;
        cumulants_[3] = calculateMoment(mu, 3U);
        cumulants_[4] = calculateMoment(mu, 4U) - 3.0*var*var;
        cumulantsCalculated_ = true;
    }

    double LegendreDistro1D::calculateMoment(const double mu,
                                             const unsigned power) const
    {
        const unsigned maxdeg = 3U + allCoeffs_.size();
        const unsigned nRule = GaussLegendreQuadrature::minimalExactRule(maxdeg);
        const GaussLegendreQuadrature glq(nRule);
        return glq.integrate(UnscaledMomentFunctor1D(*this, mu, power), -1.0, 1.0);
    }

    LegendreDistro1D::LegendreDistro1D(
        const double location, const double scale,
        const std::vector<double>& coeffs,
        const bool validatePositivity)
        : AbsLocationScaleFamily(location, scale),
          cumulantsCalculated_(false)
    {
        setupCoeffs(coeffs.empty() ? (const double*)0 : &coeffs[0],
                    coeffs.size());
        checkPositive(validatePositivity);
    }

    double LegendreDistro1D::unscaledDensity(const double x) const
    {
        if (x < -1.0 || x > 1.0)
            return 0.0;
        else
            return poly_.series(&allCoeffs_[0], allCoeffs_.size()-1U, x);
    }

    double LegendreDistro1D::unscaledDensityDerivative(const double x) const
    {
        if (x < -1.0 || x > 1.0 || derivCoeffs_.empty())
            return 0.0;
        else
            return poly_.series(&derivCoeffs_[0], derivCoeffs_.size()-1U, x);
    }

    double LegendreDistro1D::unscaledCdf(const double x) const
    {
        if (x <= -1.0)
            return 0.0;
        else if (x >= 1.0)
            return 1.0;
        else
            return poly_.series(&integCoeffs_[0], integCoeffs_.size()-1, x);
    }

    double LegendreDistro1D::unscaledExceedance(const double x) const
    {
        if (x <= -1.0)
            return 1.0;
        else if (x >= 1.0)
            return 0.0;
        else
            return 1.0 - unscaledCdf(x);
    }

    double LegendreDistro1D::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::LegendreDistro1D::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return -1.0;
        if (r1 == 1.0)
            return 1.0;
        double q;
        const bool status = findRootUsingBisections(
            UnscaledCdfFunctor1D(*this), r1, -1.0, 1.0, 2.0*DBL_EPSILON, &q);
        assert(status);
        return q;
    }

    double LegendreDistro1D::unscaledInvExceedance(const double r1) const
    {
        return unscaledQuantile(1.0 - r1);
    }

    double LegendreDistro1D::unscaledCumulant(const unsigned n) const
    {
        if (n > 4U) throw std::invalid_argument(
            "In ase::LegendreDistro1D::unscaledCumulant: "
            "only four leading cumulants are implemented");
        if (!cumulantsCalculated_)
            calculateCumulants();
        return cumulants_[n];
    }

    double LegendreDistro1D::unscaledDescentDelta(
        bool /* isToTheRight */, double /* deltaLnL */) const
    {
        throw std::runtime_error("In ase::LegendreDistro1D::unscaledDescentDelta: "
                                 "this method is not implemented");
        return 0.0;
    }

    double LegendreDistro1D::unscaledMode() const
    {
        throw std::runtime_error("In ase::LegendreDistro1D::unscaledMode: "
                                 "this method is not implemented");
        return 0.0;
    }
}
