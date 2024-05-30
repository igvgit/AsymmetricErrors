#include <cmath>
#include <utility>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/DistributionModels1D.hh"
#include "ase/kullbackLeiblerDivergence.hh"

using namespace ase;

namespace {
    double gaussKL(const double mu1, const double sig1,
                   const double mu2, const double sig2)
    {
        const double sig12 = sig1*sig1;
        const double sig22 = sig2*sig2;
        const double mudel = mu1 - mu2;
        return (mudel*mudel + sig12 - sig22)/(2.0*sig22) + log(sig2/sig1);
    }

    TEST(kullbackLeiblerDivergence_1)
    {
        const unsigned nPt = 512;

        for (unsigned itry=0; itry<10U; ++itry)
        {
            double sig1 = test_rng() + 0.2;
            double sig2 = test_rng() + 0.2;
            if (sig1 > sig2)
                std::swap(sig1, sig2);
            const double mu1 = test_rng() - 0.5;
            const double mu2 = test_rng() - 0.5;
            const double ref = gaussKL(mu1, sig1, mu2, sig2);
            const Gaussian g1(mu1, sig1);
            const Gaussian g2(mu2, sig2);

            const double dist =  kullbackLeiblerDivergence(g1, g2, nPt);
            CHECK_CLOSE(ref, dist, 1.0e-5);

            const double xmin = mu1 - 10.0*sig1;
            const double xmax = mu1 + 10.0*sig1;
            const double dist2 =  kullbackLeiblerDivergence(
                g1, g2, xmin, xmax, 4, 2000);
            CHECK_CLOSE(ref, dist2, 1.0e-10);
        }
    }
}
