#include <cmath>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/DistributionModels1D.hh"
#include "ase/NumericalConvolution.hh"
#include "ase/GaussianConvolution.hh"
#include "ase/DiscretizedConvolution.hh"
#include "ase/FunctorTimesShiftedX.hh"

using namespace ase;
using namespace std;

namespace {
    TEST(convolve_Gaussians)
    {
        const double mu1 = 0.9435;
        const double sigma1 = 1.28258;
        const double mu2 = 0.5467;
        const double sigma2 = 0.7325;

        const Gaussian g1(mu1, sigma1);
        const Gaussian g2(mu2, sigma2);
        const GaussianConvolution conv(g1, mu2, sigma2, 512);
        const NumericalConvolution nconv(g1, g2, 1024);
        const NumericalConvolution nconv2(g1, g2, 1024, 3);

        const double muconv = mu1 + mu2;
        const double gconv = hypot(sigma1, sigma2);
        const Gaussian gexpect(muconv, gconv);

        const double xmin = muconv - 10.0*gconv;
        const double xmax = muconv + 10.0*gconv;
        const DiscretizedConvolution dconv(
            g1, g2, xmin, xmax, (xmax-xmin)*1000);
        CHECK_CLOSE(1.0, dconv.densityIntegral(), 1.0e-10);

        const unsigned nIntervals = dconv.nIntervals();
        const double h = dconv.intervalWidth();
        for (unsigned i=0; i<nIntervals; ++i)
        {
            const double x = dconv.coordinateAt(i);
            const double shift = (test_rng() - 0.5)*h;
            CHECK_EQUAL(dconv.convolvedValue(i), dconv(x + shift));
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double x = (test_rng() - 0.5)*6*gexpect.scale();
            const double expect = gexpect.density(x);
            CHECK_CLOSE(expect, conv(x), 1.0e-15);
            CHECK_CLOSE(expect, nconv(x), 1.0e-7);
            CHECK_CLOSE(expect, nconv2(x), 1.0e-7);
            CHECK_CLOSE(expect, dconv(x), 1.0e-4);
        }

        const GaussHermiteQuadrature quad(256);

        {
            const double eps = 1.0e-15;

            const double norm = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(conv, gexpect, 0.0, 0U));
            CHECK_CLOSE(1.0, norm, eps);
            
            const double munum = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(conv, gexpect, 0.0, 1U));
            CHECK_CLOSE(muconv, munum, eps);
            
            const double var = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(conv, gexpect, munum, 2U));
            CHECK_CLOSE(gconv*gconv, var, eps);
            
            const double k3 = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(conv, gexpect, munum, 3U));
            CHECK_CLOSE(0.0, k3, eps);
        }
        {
            const double eps = 1.0e-5;

            const double norm = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv, gexpect, 0.0, 0U));
            CHECK_CLOSE(1.0, norm, eps);
            
            const double munum = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv, gexpect, 0.0, 1U));
            CHECK_CLOSE(muconv, munum, eps);
            
            const double var = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv, gexpect, munum, 2U));
            CHECK_CLOSE(gconv*gconv, var, eps);
            
            const double k3 = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv, gexpect, munum, 3U));
            CHECK_CLOSE(0.0, k3, eps);
        }
        {
            const double eps = 1.0e-6;

            const double norm = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv2, gexpect, 0.0, 0U));
            CHECK_CLOSE(1.0, norm, eps);
            
            const double munum = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv2, gexpect, 0.0, 1U));
            CHECK_CLOSE(muconv, munum, eps);
            
            const double var = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv2, gexpect, munum, 2U));
            CHECK_CLOSE(gconv*gconv, var, eps);
            
            const double k3 = quad.integrateProb(muconv, gconv,
                FunctorTimesShiftedXRatio(nconv2, gexpect, munum, 3U));
            CHECK_CLOSE(0.0, k3, eps);
        }
    }
}
