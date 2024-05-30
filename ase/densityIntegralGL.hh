#ifndef ASE_DENSITYINTEGRALGL_HH_
#define ASE_DENSITYINTEGRALGL_HH_

#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    /**
    // The following function performs Gauss-Legendre quadrature of
    // the density on the interval [xmin, xmax]. Can be used to check the
    // numerical performance of the quadrature (or of the cdf calculation?)
    // by comparing the result with distro.cdf(xmax) - distro.cdf(xmin).
    */
    double densityIntegralGL(const AbsDistributionModel1D& distro,
                             double xmin, double xmax,
                             unsigned nPt, unsigned nIntervals = 1U);
}

#endif // ASE_DENSITYINTEGRALGL_HH_
