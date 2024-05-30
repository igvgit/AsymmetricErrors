#include <cmath>
#include <limits>

#include "ase/DistributionModels1D.hh"
#include "ase/Poly1D.hh"
#include "ase/specialFunctions.hh"
#include "ase/DistributionFunctors1D.hh"

#define EE_DEFAULT_SAFE_SIGMA_RANGE 3.0

static double he2(const double x)
{
    return (x - 1.0)*(x + 1.0);
}

static double he3(const double x)
{
    static const double sqr3 = sqrt(3.0);
    return x*(x - sqr3)*(x + sqr3);
}

static ase::Poly1D densityFactorPoly(const double skew)
{
    // We need to construct 1 + skew/6 He_3(x).
    // Note that He_3(x) = x^3 - 3 x.
    const double sov6 = skew/6.0;
    double coeffs[4];
    coeffs[0] = 1.0;
    coeffs[1] = -3.0*sov6;
    coeffs[2] = 0.0;
    coeffs[3] = sov6;
    return ase::Poly1D(coeffs, 3);
}

static bool isNegativeOnRange(const double skew, const double range)
{
    assert(range >= 0.0);
    if (range == 0.0 || skew == 0.0)
        return false;
    else
    {
        const ase::Poly1D& poly = densityFactorPoly(skew);
        return !(poly(0.0) > 0.0 && !poly.nRoots(-range, range));
    }
}

namespace ase {
    double EdgeworthExpansion3::classSigmaRange_ = EE_DEFAULT_SAFE_SIGMA_RANGE;

    double EdgeworthExpansion3::largestSkewAllowed(const double range)
    {
        // For |skew| >= 3.0, the density factor has 3 real roots.
        // In this case the density is no longer usable in any way.
        // For skew = 3, the root with multiplicity 1 is at -2
        // and the root with multiplicity 2 is at 1.
        if (range <= 2.0)
            return 3.0;
        else if (range == 3.0)
            return 1.0/3.0;
        else
        {
            static const double eps = 2.0*std::numeric_limits<double>::epsilon();

            // Find the skewness which gives the root at -range
            double skewMin = 0.0;
            double skewMax = 3.0;
            while (2.0*(skewMax - skewMin)/(skewMin + skewMax + eps) >= eps)
            {
                const double skew = (skewMin + skewMax)/2.0;
                const ase::Poly1D& poly = densityFactorPoly(skew);
                if (poly.nRoots(-range, 0.0))
                    skewMax = skew;
                else
                    skewMin = skew;
            }
            return skewMin;
        }
    }

    void EdgeworthExpansion3::restoreDefaultSafeSigmaRange()
    {
        classSigmaRange_ = EE_DEFAULT_SAFE_SIGMA_RANGE;
    }

    void EdgeworthExpansion3::setClassSafeSigmaRange(const double range)
    {
        classSigmaRange_ = range;
        if (classSigmaRange_ < 0.0)
            classSigmaRange_ = 0.0;
    }

    void EdgeworthExpansion3::validateRange()
    {
        if (safeRange_ < 0.0)
            safeRange_ = 0.0;
        if (isNegativeOnRange(skew_, safeRange_))
            throw std::invalid_argument(
                "In ase::EdgeworthExpansion3::validateRange: "
                "the density becomes negative inside "
                "the requested safe range");
    }

    EdgeworthExpansion3::EdgeworthExpansion3(
        const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          skew_(0.0),
          safeRange_(classSigmaRange_),
          g_(0.0, 1.0)
    {
        assert(cumulants[1] > 0.0);
        const unsigned nCumulants = cumulants.size();
        if (nCumulants > 2)
        {
            const double sigma = sqrt(cumulants[1]);
            skew_ = cumulants[2]/cumulants[1]/sigma;
        }
        validateRange();
    }

    EdgeworthExpansion3::EdgeworthExpansion3(
        const double mean, const double standardDeviation,
        const double i_skewness)
        : AbsLocationScaleFamily(mean, standardDeviation),
          skew_(i_skewness),
          safeRange_(classSigmaRange_),
          g_(0.0, 1.0)
    {
        validateRange();
    }

    double EdgeworthExpansion3::densityFactor(const double x) const
    {
        return 1.0 + skew_/6.0*he3(x);
    }

    double EdgeworthExpansion3::densityFactorDerivative(const double x) const
    {
        return skew_/2.0*(x - 1.0)*(x + 1.0);
    }

    double EdgeworthExpansion3::cdfFactor(const double x) const
    {
        return skew_/6.0*he2(x);
    }

    double EdgeworthExpansion3::unscaledDensity(const double x) const
    {
        return g_.density(x)*densityFactor(x);
    }

    double EdgeworthExpansion3::unscaledDensityDerivative(const double x) const
    {
        return g_.density(x)*densityFactorDerivative(x) +
               g_.densityDerivative(x)*densityFactor(x);
    }

    double EdgeworthExpansion3::unscaledCdf(const double x) const
    {
        return g_.cdf(x) - g_.density(x)*cdfFactor(x);
    }

    double EdgeworthExpansion3::unscaledExceedance(const double x) const
    {
        return g_.exceedance(x) + g_.density(x)*cdfFactor(x);
    }

    double EdgeworthExpansion3::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        switch (n)
        {
        case 0U:
            cum = 1.0;
            break;
        case 1U:
            break;
        case 2U:
            cum = 1.0;
            break;
        case 3U:
            cum = skew_;
            break;
        case 4U:
            break;
        default:
            throw std::invalid_argument(
                "In ase::EdgeworthExpansion3::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    double EdgeworthExpansion3::unscaledQuantile(const double r1) const
    {
        static const double x0 = inverseGaussCdf(0.0);
        static const double x1 = inverseGaussCdf(1.0);
        static const double tol = 1.0e-15;

        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::EdgeworthExpansion3::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return x0;
        if (r1 == 1.0)
            return x1;

        double q = 0.0;
        const bool status = findRootUsingBisections(
            UnscaledCdfFunctor1D(*this), r1, x0, x1, tol, &q);
        if (!status) throw std::runtime_error(
            "In ase::EdgeworthExpansion3::unscaledQuantile: "
            "root finding failed");
        return q;
    }

    double EdgeworthExpansion3::unscaledInvExceedance(const double r1) const
    {
        static const double x0 = inverseGaussCdf(0.0);
        static const double x1 = inverseGaussCdf(1.0);
        static const double tol = 1.0e-15;

        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::EdgeworthExpansion3::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return x1;
        if (r1 == 1.0)
            return x0;

        double q = 0.0;
        const bool status = findRootUsingBisections(
            UnscaledExceedanceFunctor1D(*this), r1, x0, x1, tol, &q);
        if (!status) throw std::runtime_error(
            "In ase::EdgeworthExpansion3::unscaledInvExceedance: "
            "root finding failed");
        return q;
    }
}
