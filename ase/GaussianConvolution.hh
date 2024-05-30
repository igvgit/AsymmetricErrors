#ifndef ASE_GAUSSIANCONVOLUTION_HH_
#define ASE_GAUSSIANCONVOLUTION_HH_

#include "ase/DistributionModel1DCopy.hh"
#include "ase/GaussHermiteQuadrature.hh"
#include "ase/DistributionFunctors1D.hh"

namespace ase {
    /**
    // Numerical convolution with a Gaussian density
    // using Gauss-Hermite quadratures
    */
    class GaussianConvolution
    {
    public:
        // nPt should be supported by the GaussHermiteQuadrature class
        inline GaussianConvolution(const AbsDistributionModel1D& m1,
                                   const double mean, const double sigma,
                                   const unsigned nPt)
            : f1_(m1), quad_(nPt), mean_(mean), sigma_(sigma) {}

        inline double operator()(const double x) const
        {
            const ShiftedDensityFunctor1D fcn(f1_, x, true);
            return quad_.integrateProb(mean_, sigma_, fcn);
        }

    private:
        DistributionModel1DCopy f1_;
        GaussHermiteQuadrature quad_;
        double mean_;
        double sigma_;
    };
}

#endif // ASE_GAUSSIANCONVOLUTION_HH_
