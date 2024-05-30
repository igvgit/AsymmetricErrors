#ifndef ASE_NUMERICALCONVOLUTION_HH_
#define ASE_NUMERICALCONVOLUTION_HH_

#include <vector>

#include "ase/DistributionModel1DCopy.hh"

namespace ase {
    /**
    // Numerical convolution of two densities
    // using Gauss-Legendre quadratures
    */
    class NumericalConvolution
    {
    public:
        // The convolution integration will be performed by
        // transforming to a variable in which m2 density
        // turns into Uniform(0, 1). There, the interval [0, 1]
        // will be split into "nIntervals" subintervals of
        // equal length and the Gauss-Legendre quadrature
        // (with nPt points) will be performed on each subinterval.
        // nPt should be supported by the GaussLegendreQuadrature class.
        //
        // In general, the distribution m2 should be the "narrower"
        // one between m1 and m2. If one or both densities are
        // not smooth, it makes sense to use large "nIntervals"
        // and small nPt.
        //
        NumericalConvolution(const AbsDistributionModel1D& m1,
                             const AbsDistributionModel1D& m2,
                             unsigned nPt, unsigned nIntervals = 1U);

        double operator()(double x) const;

    private:
        DistributionModel1DCopy f1_;
        std::vector<double> quantiles_;
        std::vector<long double> weights_;
        mutable std::vector<double> buf_;
        double unit_;
    };
}

#endif // ASE_NUMERICALCONVOLUTION_HH_
