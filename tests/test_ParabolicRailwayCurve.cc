#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/ParabolicRailwayCurve.hh"
#include "ase/CubicHermiteInterpolatorEG.hh"
#include "ase/DerivativeFunctors.hh"

using namespace ase;

namespace {
    TEST(ParabolicRailwayCurve_extremum)
    {
        const double eps = 1.0e-14;

        for (unsigned itry=0; itry<10000; ++itry)
        {
            double sigmaPlus = 0.0, sigmaMinus = 0.0;
            while (!(sigmaMinus > 0.0 && sigmaPlus < sigmaMinus))
            {
                sigmaPlus = test_rng() - 0.5;
                sigmaMinus = test_rng() - 0.5;
            }
            const ParabolicRailwayCurve<double> rc(sigmaPlus, sigmaMinus, 1.0, 1.0);
            CHECK(rc.secondDerivative(0.0) < 0.0);
            const auto& dfcn = DerivativeFunctor(rc);
            const CubicHermiteInterpolatorEG interp(-3.0, 3.0, 7, rc, dfcn);
            if (rc.hasExtremum())
            {
                const auto extremum = rc.extremum();
                CHECK(rc.secondDerivative(extremum.first) < 0.0);
                CHECK_CLOSE(0.0, rc.derivative(extremum.first), eps);
                CHECK_CLOSE(interp.argmax(), extremum.first, eps);
                CHECK_CLOSE(interp.maximum(), extremum.second, eps);
            }
            else
                CHECK(std::abs(interp.argmax()) == 3.0);
        }
    }

    TEST(ParabolicRailwayCurve_inverse)
    {
        const double eps = 1.0e-11;

        double inv[2];
        for (unsigned itry=0; itry<100; ++itry)
        {
            const double sigmaPlus = test_rng() - 0.5;
            const double sigmaMinus = test_rng() - 0.5;
            const double h1 = test_rng()/2.0 + 0.5;
            const double h2 = test_rng()/2.0 + 0.5;
            const ParabolicRailwayCurve<double> rc(sigmaPlus, sigmaMinus, h1, h2);

            for (unsigned i=0; i<1000; ++i)
            {
                const double x = 8.0*(test_rng() - 0.5);
                const double y = rc(x);
                const unsigned nSols = rc.inverse(y, inv);
                CHECK(nSols > 0U);

                if (nSols == 1U)
                    CHECK_CLOSE(x, inv[0], eps);
                else
                {
                    CHECK(nSols == 2U);
                    CHECK(rc.hasExtremum());
                    CHECK(inv[0] <= inv[1]);
                    CHECK(std::abs(x - inv[0]) < eps || std::abs(x - inv[1]) < eps);
                }
            }
        }
    }
}
