#include <cmath>
#include <limits>
#include <cassert>

#include "ase/Gaussian.hh"
#include "ase/specialFunctions.hh"

#define LOG2PI 1.8378770664093454836
#define SQR2PI 2.5066282746310005024
#define GAUSSIAN_ENTROPY ((LOG2PI + 1.0)/2.0)
#define TWOPIL 6.28318530717958647692528676656L

static long double ldrandom(ase::AbsRNG& g)
{
    static const long double extra = sqrt(std::numeric_limits<double>::epsilon());

    long double u = 0.0L;
    while (u <= 0.0L || u >= 1.0L)
    {
        u = g()*(1.0L + extra) - extra/2.0L;
        u += (g() - 0.5L)*extra;
    }
    return u;
}

namespace ase {
    // Initialize static members
    const double Gaussian::xmin_ = inverseGaussCdf(0.0);
    const double Gaussian::xmax_ = inverseGaussCdf(1.0);

    std::unique_ptr<Gaussian> Gaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus != sigmaMinus) throw std::invalid_argument(
            "In ase::Gaussian::fromQuantiles: "
            "can't create this object using non-equal sigmas");
        if (sigmaPlus <= 0.0) throw std::invalid_argument(
            "In ase::Gaussian::fromQuantiles: "
            "sigma parameters must be positive");
        return std::unique_ptr<Gaussian>(new Gaussian(median, sigmaPlus));
    }

    double Gaussian::unscaledEntropy() const
    {
        return GAUSSIAN_ENTROPY;
    }

    Gaussian::Gaussian(const double mu, const double sigma)
        : AbsLocationScaleFamily(mu, sigma)
    {
    }

    Gaussian::Gaussian(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1)))
    {
        assert(cumulants[1] > 0.0);
    }

    double Gaussian::unscaledDensity(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else
            return exp(-x*x/2.0)/SQR2PI;
    }

    double Gaussian::unscaledDensityDerivative(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else
            return -x*exp(-x*x/2.0)/SQR2PI;
    }

    double Gaussian::unscaledCdf(const double x) const
    {
        if (x <= xmin_)
            return 0.0;
        if (x >= xmax_)
            return 1.0;
        if (x < 0.0)
            return erfc(-x/M_SQRT2)/2.0;
        else
            return (1.0 + erf(x/M_SQRT2))/2.0;
    }

    double Gaussian::unscaledExceedance(const double x) const
    {
        if (x <= xmin_)
            return 1.0;
        if (x >= xmax_)
            return 0.0;
        if (x > 0.0)
            return erfc(x/M_SQRT2)/2.0;
        else
            return (1.0 - erf(x/M_SQRT2))/2.0;
    }

    double Gaussian::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::Gaussian::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        return inverseGaussCdf(r1);
    }

    double Gaussian::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::Gaussian::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        return -inverseGaussCdf(r1);
    }

    double Gaussian::unscaledCumulant(const unsigned n) const
    {
        switch (n)
        {
        case 0U:
        case 2U:
            return 1.0;
        default:
            return 0.0;
        }
    }

    double Gaussian::unscaledRandom(AbsRNG& gen) const
    {
        const long double r1 = ldrandom(gen);
        const long double r2 = ldrandom(gen);
        return sqrtl(-2.0L*logl(r1))*sinl(TWOPIL*(r2-0.5L));
    }

    double Gaussian::unscaledDescentDelta(bool /* isToTheRight */,
                                          const double deltaLnL) const
    {
        assert(deltaLnL > 0.0);
        return sqrt(2.0*deltaLnL);
    }
}
