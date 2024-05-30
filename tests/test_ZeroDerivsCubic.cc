#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/ZeroDerivsCubic.hh"

using namespace ase;

namespace {
    TEST(ZeroDerivsCubic_1)
    {
        const double eps = 3.0e-8;

        for (unsigned itry=0; itry<100; ++itry)
        {
            const double x = test_rng() - 0.5;
            const double y0 = test_rng() - 0.5;
            const double yx = test_rng() - 0.5;

            if (x)
            {
                const ZeroDerivsCubic cubic(y0, x, yx);

                CHECK_CLOSE(y0, cubic(0.0), eps);
                CHECK_CLOSE(yx, cubic(x), eps);
                CHECK_CLOSE(0.0, cubic.derivative(x), eps);
                CHECK_CLOSE(0.0, cubic.secondDerivative(x), eps);
            }
        }
    }
}
