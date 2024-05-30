#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/statUtils.hh"

using namespace ase;
using namespace std;

namespace {
    TEST(cumulantConversions)
    {
        const long double eps = 1.0e-15;

        long double cums[4];
        long double moms[4];
        long double cums2[4];

        for (unsigned itry=0; itry<100; ++itry)
        {
            cums[0] = test_rng() - 0.5;
            cums[1] = test_rng() + 0.5;
            cums[2] = test_rng() - 0.5;
            cums[3] = test_rng() - 0.5;

            cumulantsToMoments(cums, moms, 4U);
            momentsToCumulants(moms, cums2, 4U);

            for (unsigned i=0; i<4U; ++i)
                CHECK_CLOSE(cums[i], cums2[i], eps);
        }
    }
}
