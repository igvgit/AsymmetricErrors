#include <cmath>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/HermiteProbOrthoPoly.hh"
#include "ase/Poly1D.hh"

#define SQR2PIL 2.506628274631000502415765L
#define SQR2L 1.414213562373095048801689L

using namespace ase;

namespace {
    long double mono_integ_0(const long double x)
    {
        return (1 + erfl(x/SQR2L))/2;
    }

    long double mono_integ_1(const long double x)
    {
        return -(1/(expl(x*x/2)*SQR2PIL));
    }

    long double mono_integ_2(const long double x)
    {
        return 0.5L - x/(expl(x*x/2)*SQR2PIL) + erfl(x/SQR2L)/2;
    }

    long double mono_integ_3(const long double x)
    {
        const long double x2 = x*x;
        return -((2 + x2)/(expl(x2/2)*SQR2PIL));
    }

    long double mono_integ_4(const long double x)
    {
        const long double x2 = x*x;
        return -((x*(3 + x2))/(expl(x2/2)*SQR2PIL)) + (3*(1 + erfl(x/SQR2L)))/2;
    }

    long double mono_integ_5(const long double x)
    {
        const long double x2 = x*x;
        return -((8 + 4*x2 + x2*x2)/(expl(x2/2)*SQR2PIL));
    }

    TEST(mono_integ)
    {
        const long double eps = 1.0e-17;
        long double c[6];

        c[0] = 1;
        c[1] = -3;
        c[2] = 5;
        c[3] = -0.5L;
        c[4] = 1/3.0L;
        c[5] = 7;
        const long double x = 1.2345L;
        long double sum =
              c[0]*mono_integ_0(x)
            + c[1]*mono_integ_1(x)
            + c[2]*mono_integ_2(x)
            + c[3]*mono_integ_3(x)
            + c[4]*mono_integ_4(x)
            + c[5]*mono_integ_5(x);
        CHECK_CLOSE(-15.76871630979080724976454L, sum, eps);

        c[0] = 2;
        c[1] = -0.5L;
        c[2] = 7;
        c[3] = -11;
        c[4] = 13;
        c[5] = -0.2L;
        sum = c[0]*mono_integ_0(x)
            + c[1]*mono_integ_1(x)
            + c[2]*mono_integ_2(x)
            + c[3]*mono_integ_3(x)
            + c[4]*mono_integ_4(x)
            + c[5]*mono_integ_5(x);
        CHECK_CLOSE(35.58618682416268964585591L, sum, eps);
    }

    TEST(HermiteProbOrthoPoly_values)
    {
        const long double eps = 1.0e-17;

        const HermiteProbOrthoPoly he;

        long double x = 1/3.0L;
        CHECK_CLOSE(he.weight(x), expl(-x*x/2)/SQR2PIL, eps);
        CHECK_CLOSE(he.poly(0, x), 1.0L, eps);
        CHECK_CLOSE(he.poly(1, x), 1/3.0L, eps);
        CHECK_CLOSE(he.poly(2, x), -0.62853936105470891058L, eps);
        CHECK_CLOSE(he.poly(3, x), -0.39312798340964586761L, eps);
        CHECK_CLOSE(he.poly(4, x), 0.47880972338354304389L, eps);
    }

    TEST(HermiteProbOrthoPoly_series)
    {
        const long double eps = 1.0e-16;

        const unsigned maxdeg = 5U;
        long double monoCoeffs[maxdeg + 1U];
        long double seriesCoeffs[maxdeg + 1U];

        HermiteProbOrthoPoly he;

        for (unsigned i=0; i<10; ++i)
        {
            for (unsigned deg=0; deg<=maxdeg; ++deg)
                monoCoeffs[deg] = test_rng() - 0.5;

            const Poly1D mono(monoCoeffs, maxdeg);
            he.calculateCoeffs(mono, seriesCoeffs, maxdeg);

            for (unsigned j=0; j<10; ++j)
            {
                const long double x = 6.0*(test_rng() - 0.5);

                const long double mValue = mono(x);
                const long double ser = he.series(seriesCoeffs, maxdeg, x);
                CHECK_CLOSE(mValue, ser, eps);

                CHECK_CLOSE(mono_integ_0(x), he.weightIntegral(x), eps);

                const long double integ = he.weightedSeriesIntegral(
                    seriesCoeffs, maxdeg, x);
                const long double ref =
                      monoCoeffs[0]*mono_integ_0(x)
                    + monoCoeffs[1]*mono_integ_1(x)
                    + monoCoeffs[2]*mono_integ_2(x)
                    + monoCoeffs[3]*mono_integ_3(x)
                    + monoCoeffs[4]*mono_integ_4(x)
                    + monoCoeffs[5]*mono_integ_5(x);
                CHECK_CLOSE(ref, integ, eps);
            }
        }
    }
}
