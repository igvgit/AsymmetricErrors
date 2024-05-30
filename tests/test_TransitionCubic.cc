#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/TransitionCubic.hh"
#include "ase/CubicHermiteInterpolatorEG.hh"
#include "ase/DerivativeFunctors.hh"

using namespace ase;

namespace {
    TEST(TransitionCubic_extremum)
    {
        const double eps = 1.0e-14;

        for (unsigned itry=0; itry<10000; ++itry)
        {
            const double x0 = test_rng() - 0.5;
            double h = 0.0;
            while (std::abs(h) < 0.01)
                h = test_rng() - 0.5;
            const double v0 = test_rng() - 0.5;
            const double d0 = test_rng() - 0.5;
            const double sd = test_rng() - 1.0;

            const TransitionCubic<double> tc(x0, h, v0, d0, sd);
            const auto& dfcn = DerivativeFunctor(tc);
            double xmin = x0;
            double xmax = x0 + h;
            if (xmin > xmax)
                std::swap(xmin, xmax);
            const CubicHermiteInterpolatorEG interp(xmin, xmax, 2, tc, dfcn);
            if (tc.hasExtremum())
            {
                const auto extremum = tc.extremum();
                CHECK_CLOSE(interp.argmax(), extremum.first, eps);
                CHECK_CLOSE(interp.maximum(), extremum.second, eps);
            }
            else
                CHECK(interp.argmax() == xmin || interp.argmax() == xmax);
        }
    }
}
