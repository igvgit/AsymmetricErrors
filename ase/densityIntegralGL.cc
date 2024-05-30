#include "ase/DistributionFunctors1D.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/densityIntegralGL.hh"

namespace ase {
    double densityIntegralGL(const AbsDistributionModel1D& distro,
                             const double xmin, const double xmax,
                             const unsigned nPt, const unsigned nIntervals)
    {
        const GaussLegendreQuadrature glq(nPt);
        return glq.integrate(DensityFunctor1D(distro), xmin, xmax, nIntervals);
    }
}
