#ifndef ASE_KULLBACKLEIBLERDIVERGENCE_HH_
#define ASE_KULLBACKLEIBLERDIVERGENCE_HH_

#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    /**
    // The following function calculates the average value of the
    // log(m1.density()/m2.density()) w.r.t. distribution m1.
    // The calculation is performed by transforming to a variable
    // in which the m1 density turns into Uniform(0, 1). There, the
    // interval [0, 1] will be split into "nIntervals" subintervals of
    // equal length and the Gauss-Legendre quadrature (with nPt points)
    // will be performed on each subinterval. nPt should be supported
    // by the GaussLegendreQuadrature class. Note that the precision
    // of the result will strongly depend on how well, in the cdf space,
    // log(m1.density()/m2.density()) can be represented by polynomials.
    */
    double kullbackLeiblerDivergence(const AbsDistributionModel1D& m1,
                                     const AbsDistributionModel1D& m2,
                                     unsigned nPt, unsigned nIntervals = 1U);

    /**
    // The following function calculates the Kullback-Leibler divergence
    // by performing Gauss-Legendre quadrature in the original variable
    // for which the distributions are defined. The integral is calculated
    // on the interval [xmin, xmax]. This interval should be chosen
    // in such a way that only a small tail (if any) of the m1 distribution
    // remains outside of it.
    */
    double kullbackLeiblerDivergence(const AbsDistributionModel1D& m1,
                                     const AbsDistributionModel1D& m2,
                                     double xmin, double xmax,
                                     unsigned nPt, unsigned nIntervals = 1U);
}

#endif // ASE_KULLBACKLEIBLERDIVERGENCE_HH_
