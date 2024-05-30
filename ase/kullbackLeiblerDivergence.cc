#include <cmath>
#include <cassert>
#include <stdexcept>

#include "ase/GaussLegendreQuadrature.hh"
#include "ase/kullbackLeiblerDivergence.hh"

namespace {
    class DensityLogRatio
    {
    public:
        inline DensityLogRatio(const ase::AbsDistributionModel1D& m1,
                               const ase::AbsDistributionModel1D& m2)
            : m1_(m1), m2_(m2) {}

        inline double operator()(const double cdf) const
        {
            const double x = m1_.quantile(cdf);
            const double d1 = m1_.density(x);
            assert(d1 > 0.0);
            const double d2 = m2_.density(x);
            assert(d2 >= 0.0);
            if (d2 == 0.0) throw std::invalid_argument(
                "In DensityLogRatio: zero density value encountered");
            return log(d1/d2);
        }

    private:
        const ase::AbsDistributionModel1D& m1_;
        const ase::AbsDistributionModel1D& m2_;
    };

    class KLIntegrand
    {
    public:
        inline KLIntegrand(const ase::AbsDistributionModel1D& m1,
                           const ase::AbsDistributionModel1D& m2)
            : m1_(m1), m2_(m2) {}

        inline double operator()(const double x) const
        {
            const double d1 = m1_.density(x);
            assert(d1 >= 0.0);
            if (d1 == 0.0)
                return 0.0;
            const double d2 = m2_.density(x);
            assert(d2 >= 0.0);
            if (d2 == 0.0) throw std::invalid_argument(
                "In KLIntegrand: zero density value encountered");
            return d1*log(d1/d2);
        }

    private:
        const ase::AbsDistributionModel1D& m1_;
        const ase::AbsDistributionModel1D& m2_;
    };
}

namespace ase {
    double kullbackLeiblerDivergence(const AbsDistributionModel1D& m1,
                                     const AbsDistributionModel1D& m2,
                                     const unsigned nPt,
                                     const unsigned nIntervals)
    {
        const GaussLegendreQuadrature glq(nPt);
        const DensityLogRatio lnRatio(m1, m2);
        return glq.integrate(lnRatio, 0.0, 1.0, nIntervals);
    }

    double kullbackLeiblerDivergence(const AbsDistributionModel1D& m1,
                                     const AbsDistributionModel1D& m2,
                                     const double xmin, const double xmax,
                                     const unsigned nPt,
                                     const unsigned nIntervals)
    {
        const GaussLegendreQuadrature glq(nPt);
        const KLIntegrand kl(m1, m2);
        return glq.integrate(kl, xmin, xmax, nIntervals);
    }
}
