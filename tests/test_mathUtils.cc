#include <cmath>
#include <limits>
#include <algorithm>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/mathUtils.hh"
#include "ase/Poly1D.hh"

using namespace ase;
using namespace std;

namespace {
    TEST(hermiteSeriesDerivative)
    {
        const double eps = 1.0e-10;
        const unsigned maxdeg = 6;
        const long double h = powl(numeric_limits<long double>::epsilon(), 1.0L/3);

        long double coeffs[maxdeg + 1U];
        long double derivCoeffs[maxdeg];

        for (unsigned itry=0; itry<100; ++itry)
        {
            for (unsigned i=0; i<=maxdeg; ++i)
                coeffs[i] = test_rng() - 0.5;
            hermiteSeriesDerivative(coeffs, maxdeg, derivCoeffs);

            for (unsigned i=0; i<10; ++i)
            {
                const long double x = test_rng() - 0.5;
                const volatile long double xplus = x + h;
                const volatile long double xminus = x - h;
                const long double nder = (hermiteSeriesSumProb(coeffs, maxdeg, xplus) -
                                          hermiteSeriesSumProb(coeffs, maxdeg, xminus))/
                                          (xplus - xminus);
                const long double der2 = hermiteSeriesSumProb(derivCoeffs, maxdeg-1U, x);
                CHECK_CLOSE(nder, der2, eps);
            }
        }
    }

    TEST(hermiteSeriesCoeffs)
    {
        const double eps = 1.0e-14;

        const unsigned maxdeg = 10;
        long double coeffs[maxdeg + 1];
        long double hermCoeffs[maxdeg + 1];

        for (unsigned itry=0; itry<100; ++itry)
        {
            const unsigned deg = test_rng()*(maxdeg + 0.9999);
            for (unsigned i=0; i<=deg; ++i)
                coeffs[i] = test_rng() - 0.5;
            const Poly1D poly(coeffs, deg);
            hermiteSeriesCoeffs(poly, hermCoeffs, deg);

            for (unsigned i=0; i<10; ++i)
            {
                const long double x = test_rng() - 0.5;
                const long double value = hermiteSeriesSumProb(hermCoeffs, deg, x);
                CHECK_CLOSE(poly(x), value, eps);
            }
        }
    }

    TEST(hermiteSeriesRoots)
    {
        const long double eps = 1.0e-5;
        const unsigned maxdeg = 7;
        const unsigned ntry = 10000;
        const long double a = 0.25;
        const long double b = 0.75;
        
        long double rootBuf[maxdeg], inRoots[maxdeg];
        long double hermCoeffs[maxdeg + 1U];

        for (unsigned i=0; i<ntry; ++i)
        {
            const unsigned deg1 = test_rng()*(maxdeg + 0.99);
            unsigned rootcount = 0U;
            Poly1D poly(1.0L);
            for (unsigned i=0; i<deg1; ++i)
            {
                long double root = test_rng();
                if (root > a && root < b)
                    inRoots[rootcount++] = root;
                poly *= Poly1D::monicDeg1(-root);
            }
            std::sort(inRoots, inRoots+rootcount);
            hermiteSeriesCoeffs(poly, hermCoeffs, poly.deg());
            const unsigned nFound = hermiteSeriesRoots(
                hermCoeffs, poly.deg(), a, b, rootBuf);
            CHECK(nFound <= poly.deg());
            CHECK_EQUAL(nFound, rootcount);
            for (unsigned i=0; i<nFound; ++i)
                CHECK_CLOSE(inRoots[i], rootBuf[i], eps);
        }
    }

    TEST(hermiteSeriesWithDeriv)
    {
        const double eps = 1.0e-15;
        const unsigned maxdeg = 6;

        long double coeffs[maxdeg + 1U];
        long double derivCoeffs[maxdeg];
        long double sDerivCoeffs[maxdeg - 1U];

        for (unsigned itry=0; itry<100; ++itry)
        {
            for (unsigned i=0; i<=maxdeg; ++i)
                coeffs[i] = test_rng() - 0.5;
            hermiteSeriesDerivative(coeffs, maxdeg, derivCoeffs);
            hermiteSeriesDerivative(derivCoeffs, maxdeg-1, sDerivCoeffs);

            long double value, d, sd;
            for (unsigned i=0; i<10; ++i)
            {
                const long double x = test_rng() - 0.5;
                hermiteSeriesWithDeriv(coeffs, maxdeg, x,
                                       &value, &d, &sd);
                const long double v0 = hermiteSeriesSumProb(coeffs, maxdeg, x);
                const long double d0 = hermiteSeriesSumProb(derivCoeffs, maxdeg-1, x);
                const long double sd0 = hermiteSeriesSumProb(sDerivCoeffs, maxdeg-2, x);
                CHECK_CLOSE(v0, value, eps);
                CHECK_CLOSE(d0, d, eps);
                CHECK_CLOSE(sd0, sd, eps);
            }
        }
    }

    TEST(solveCubic)
    {
        const double eps = 1.0e-10;
        double roots[3], foundRoots[3];
        
        for (unsigned itry=0; itry<100; ++itry)
        {
            const double a = test_rng() - 0.5;
            const double b = test_rng() - 0.5;
            const double c = test_rng() - 0.5;
            const double p = -(a + b + c);
            const double q = a*b + a*c + b*c;
            const double r = -(a*b*c);

            roots[0] = a;
            roots[1] = b;
            roots[2] = c;
            sort(roots, roots+3);

            const unsigned nRoots = solveCubic(p, q, r, foundRoots);
            CHECK_EQUAL(nRoots, 3U);
            if (nRoots == 3U)
            {
                sort(foundRoots, foundRoots+3);
                for (unsigned i=0; i<3; ++i)
                    CHECK_CLOSE(roots[i], foundRoots[i], eps);
            }
        }
    }
}
