#include <algorithm>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/mathUtils.hh"
#include "ase/Poly1D.hh"

using namespace ase;

namespace {
    TEST(Poly1D_derivative_integral)
    {
        const long double eps = 1.0e-18;

        const unsigned maxdeg = 10;
        long double coeffs[maxdeg + 1];
        const unsigned ntry = 100;
        const unsigned ncheck = 10;

        for (unsigned deg=0; deg<=maxdeg; ++deg)
        {
            for (unsigned itry=0; itry<ntry; ++itry)
            {
                for (unsigned i=0; i<=deg; ++i)
                    coeffs[i] = test_rng();
                const Poly1D poly(coeffs, deg);
                const Poly1DShifted fcn(poly, 0.0L);
                const Poly1D& deriv1 = poly.derivative();
                const Poly1D& integ = poly.integral(7.0);
                const Poly1D& deriv2 = integ.derivative();
                for (unsigned icheck=0; icheck<ncheck; ++icheck)
                {
                    const double x = 2.0*test_rng() - 1.0;
                    long double value, deriv;
                    polyAndDeriv(coeffs, deg, x, &value, &deriv);
                    CHECK_CLOSE(value, poly(x), eps);
                    CHECK_CLOSE(value, fcn(x), eps);
                    CHECK_CLOSE(value, deriv2(x), eps);
                    CHECK_CLOSE(deriv, deriv1(x), eps);
                }
            }
        }

        coeffs[0] = 2;
        coeffs[1] = 3;
        coeffs[2] = 5;
        Poly1D poly(coeffs, 2);

        coeffs[0] = 7;
        coeffs[1] = 2;
        coeffs[2] = 3/2.0L;
        coeffs[3] = 5/3.0L;
        Poly1D poly2(coeffs, 3);
        CHECK(poly.integral(7).isClose(poly2, eps));
    }

    TEST(Poly1D_add_subtract)
    {
        const double eps = 1.0e-17;

        const unsigned maxdeg = 10;
        long double c1[maxdeg + 1];
        long double c2[maxdeg + 1];
        const unsigned ntry = 1000;
        const unsigned ncheck = 10;

        for (unsigned i=0; i<ntry; ++i)
        {
            const unsigned deg1 = test_rng()*(maxdeg + 0.99);
            const unsigned deg2 = test_rng()*(maxdeg + 0.99);
            for (unsigned i=0; i<=deg1; ++i)
                c1[i] = test_rng();
            for (unsigned i=0; i<=deg2; ++i)
                c2[i] = test_rng();
            Poly1D p1(c1, deg1);
            Poly1D p2(c2, deg2);
            Poly1D psum1(p1 + p2);
            Poly1D pdiff1(p1 - p2);
            Poly1D psum2(p1);
            psum2 += p2;
            Poly1D pdiff2(p1);
            pdiff2 -= p2;
            Poly1D pdiff3(-pdiff2);
            Poly1D pdiff4(+pdiff2);
            Poly1D psum3(p1 + 13.0);
            Poly1D pdiff5(p1 - 17.0);

            for (unsigned icheck=0; icheck<ncheck; ++icheck)
            {
                const long double x = 2.0*test_rng() - 1.0;
                const long double v1 = polySeriesSum(c1, deg1, x);
                const long double v2 = polySeriesSum(c2, deg2, x);
                CHECK_CLOSE(v1+v2, psum1(x), eps);
                CHECK_CLOSE(v1+v2, psum2(x), eps);
                CHECK_CLOSE(v1-v2, pdiff1(x), eps);
                CHECK_CLOSE(v1-v2, pdiff2(x), eps);
                CHECK_CLOSE(v1-v2, pdiff4(x), eps);
                CHECK_CLOSE(v2-v1, pdiff3(x), eps);
                CHECK_CLOSE(v1 + 13.0, psum3(x), eps);
                CHECK_CLOSE(v1 - 17.0, pdiff5(x), eps);
            }    
        }
    }

    TEST(Poly1D_mul_div)
    {
        const long double eps = 2.0e-17;

        const unsigned maxdeg = 10;
        long double c1[maxdeg + 1];
        long double c2[maxdeg + 1];
        const unsigned ntry = 1000;
        const unsigned ncheck = 10;

        for (unsigned i=0; i<ntry; ++i)
        {
            const unsigned deg1 = test_rng()*(maxdeg + 0.99);
            const unsigned deg2 = test_rng()*(maxdeg + 0.99);
            for (unsigned i=0; i<=deg1; ++i)
                c1[i] = test_rng();
            for (unsigned i=0; i<=deg2; ++i)
                c2[i] = test_rng();
            c2[deg2] += 0.5;

            Poly1D p1(c1, deg1);
            Poly1D p2(c2, deg2);
            const long double c3 = test_rng();
            Poly1D p3(p1*c3);
            Poly1D p4(p1*p2);
            Poly1D p8(p1);
            p8 *= p2;

            Poly1D p5(p1/p2);
            Poly1D p6(p1%p2);
            if (p1.deg() >= p2.deg())
            {
                CHECK(p1.deg() == p5.deg() + p2.deg());
                if (p2.deg())
                    CHECK(p6.deg() < p2.deg());
                else
                    CHECK(p6.isNull());
            }
            else
            {
                CHECK(p5.isNull());
                CHECK(p6 == p1);
            }

            Poly1D p7(p5*p2 + p6);
            CHECK(p7.isClose(p1, eps));

            for (unsigned icheck=0; icheck<ncheck; ++icheck)
            {
                const long double x = 2.0*test_rng() - 1.0;
                const long double v1 = polySeriesSum(c1, deg1, x);
                const long double v2 = polySeriesSum(c2, deg2, x);
                CHECK_CLOSE(v1*c3, p3(x), eps);
                CHECK_CLOSE(v1*v2, p4(x), eps);
                CHECK_CLOSE(v1*v2, p8(x), eps);
                CHECK_CLOSE(v1, p5(x)*p2(x) + p6(x), eps);
           }
        }
    }

    TEST(Poly1D_nroots_on_interval)
    {
        const unsigned maxdeg = 10;
        const unsigned ntry = 10000;
        const long double a = 0.25;
        const long double b = 0.75;

        for (unsigned i=0; i<ntry; ++i)
        {
            const unsigned deg1 = test_rng()*(maxdeg + 0.99);
            unsigned rootcount = 0U;
            Poly1D poly(1.0L);
            for (unsigned i=0; i<deg1; ++i)
            {
                long double root = test_rng();
                if (root > a && root <= b)
                    ++rootcount;
                poly *= Poly1D::monicDeg1(-root);
            }
            CHECK_EQUAL(rootcount, poly.nRoots(a, b));
        }
    }

    TEST(Poly1D_roots_on_interval)
    {
        const long double eps = 1.0e-6;
        const unsigned maxdeg = 10;
        const unsigned ntry = 10000;
        const long double a = 0.25;
        const long double b = 0.75;
        
        long double rootBuf[maxdeg], inRoots[maxdeg];
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
            const unsigned nFound = poly.findRoots(a, b, rootBuf);
            CHECK(nFound <= poly.deg());
            CHECK_EQUAL(nFound, rootcount);
            for (unsigned i=0; i<nFound; ++i)
                CHECK_CLOSE(inRoots[i], rootBuf[i], eps);
        }
    }

    TEST(Poly1D_monic)
    {
        const double eps = 1.0e-17;

        Poly1D mon0(Poly1D::monicDeg0());
        Poly1D mon1(Poly1D::monicDeg1(5.0));
        Poly1D mon2(Poly1D::monicDeg2(7.0, 13.0));

        for (unsigned i=0; i<10; ++i)
        {
            const long double x = test_rng();
            CHECK_EQUAL(1.0L, mon0(x));
            CHECK_CLOSE(x + 5.0, mon1(x), eps);
            CHECK_CLOSE(x*x + 7.0*x + 13.0, mon2(x), eps);
        }
    }

    TEST(Poly1D_setCoefficient)
    {
        Poly1D p1(13, 5.0);
        Poly1D p2;
        p2.setCoefficient(13, 5.0);
        CHECK(p1 == p2);
    }
}
