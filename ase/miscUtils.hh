#ifndef ASE_MISCUTILS_HH_
#define ASE_MISCUTILS_HH_

#include <cmath>
#include <limits>
#include <cassert>

namespace ase {
    unsigned printed_unsigned_width(unsigned u);

    template<typename Real>
    void assert_approx_equal(const Real x, const Real y)
    {
        static const Real eps = 2.0*std::numeric_limits<Real>::epsilon();
        static const Real sqreps = std::sqrt(eps);

        const Real scale = (std::abs(x) + std::abs(y))/2.0;
        const Real delta = std::abs(x - y);
        assert(delta/(scale + sqreps) <= eps);
    }
}

#endif // ASE_MISCUTILS_HH_
