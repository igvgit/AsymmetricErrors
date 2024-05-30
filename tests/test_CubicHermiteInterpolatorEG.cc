#include <cmath>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/CubicHermiteInterpolatorEG.hh"
#include "ase/LikelihoodCurveCopy.hh"
#include "ase/Poly1D.hh"

using namespace ase;
using namespace std;

namespace {
    double get_x(const double xmin, const double xmax,
                 const unsigned cycle)
    {
        if (cycle == 0U)
            return xmin;
        else if (cycle == 1U)
            return xmax;
        else
            return xmin + (xmax - xmin)*test_rng();
    }

    class Quadratic
    {
    public:
        inline Quadratic(const double a, const double peakX, const double peakY)
            : a_(a), peakX_(peakX), peakY_(peakY) {}

        inline double operator()(const double x) const
        {
            const double del = x - peakX_;
            return a_*del*del + peakY_;
        }

    private:
        double a_;
        double peakX_;
        double peakY_;
    };

    class QuadraticDerivative
    {
    public:
        inline QuadraticDerivative(const double a, const double peakX)
            : a_(a), peakX_(peakX) {}

        inline double operator()(const double x) const
        {
            const double del = x - peakX_;
            return 2.0*a_*del;
        }

    private:
        double a_;
        double peakX_;
    };
    
    TEST(CubicHermiteInterpolatorEG_1)
    {
        const double eps = 1.0e-12;
        const double xmin = 1.0;
        const double xmax = 2.0;
        const unsigned nPoints = 5;
        
        double coeffs[4];
        for (unsigned icycle=0; icycle<10; ++icycle)
        {
            for (unsigned i=0; i<4; ++i)
                coeffs[i] = test_rng();

            const Poly1D poly(coeffs, 3);
            const Poly1D& deriv = poly.derivative();

            const CubicHermiteInterpolatorEG cg(xmin, xmax, nPoints, poly, deriv);
            for (unsigned itest=0; itest<10; ++itest)
            {
                const double x = get_x(xmin, xmax, itest);
                CHECK_CLOSE(poly(x), cg(x), eps);
                CHECK_CLOSE(deriv(x), cg.derivative(x), eps);
            }
        }
    }

    TEST(CubicHermiteInterpolatorEG_2)
    {
        const double eps = 1.0e-10;
        const double xmin = -1.5;
        const double xmax = 2.5435;
        const unsigned nPoints = 5;
        
        double coeffs[4];
        for (unsigned icycle=0; icycle<10; ++icycle)
        {
            for (unsigned i=0; i<4; ++i)
                coeffs[i] = test_rng();

            const Poly1D poly(coeffs, 3);
            const Poly1D& deriv = poly.derivative();

            const CubicHermiteInterpolatorEG cg(xmin, xmax, nPoints, poly, 1.0e-5);
            for (unsigned itest=0; itest<10; ++itest)
            {
                const double x = get_x(xmin, xmax, itest);
                CHECK_CLOSE(poly(x), cg(x), eps);
                CHECK_CLOSE(deriv(x), cg.derivative(x), eps);
            }
        }
    }

    TEST(CubicHermiteInterpolatorEG_3)
    {
        const double eps = 1.0e-10;
        const double xmin = -1.5;
        const double xmax = 2.5435;
        const unsigned nPoints = 5;
        
        double coeffs[4];
        vector<double> values(nPoints), derivs(nPoints);
        EquidistantGrid grid(nPoints, xmin, xmax);

        for (unsigned icycle=0; icycle<10; ++icycle)
        {
            for (unsigned i=0; i<4; ++i)
                coeffs[i] = test_rng();

            const Poly1D poly(coeffs, 3);
            const Poly1D& deriv = poly.derivative();
            for (unsigned i=0; i<nPoints; ++i)
            {
                const double x = grid.coordinate(i);
                values[i] = poly(x);
                derivs[i] = deriv(x);
            }

            const CubicHermiteInterpolatorEG cg(xmin, xmax, values, derivs);
            for (unsigned itest=0; itest<10; ++itest)
            {
                const double x = get_x(xmin, xmax, itest);
                CHECK_CLOSE(poly(x), cg(x), eps);
                CHECK_CLOSE(deriv(x), cg.derivative(x), eps);
            }
        }
    }

    TEST(CubicHermiteInterpolatorEG_4)
    {
        const double eps = 1.0e-7;
        const double xmin = -1.0;
        const double xmax = 1.0;
        const unsigned nPoints = 10001;

        double coeffs[4];
        vector<double> values(nPoints);
        EquidistantGrid grid(nPoints, xmin, xmax);

        for (unsigned icycle=0; icycle<10; ++icycle)
        {
            for (unsigned i=0; i<4; ++i)
                coeffs[i] = test_rng();

            const Poly1D poly(coeffs, 3);
            const Poly1D& deriv = poly.derivative();
            for (unsigned i=0; i<nPoints; ++i)
            {
                const double x = grid.coordinate(i);
                values[i] = poly(x);
            }

            const CubicHermiteInterpolatorEG cg(xmin, xmax, values);
            for (unsigned itest=0; itest<10; ++itest)
            {
                const double x = get_x(xmin, xmax, itest);
                CHECK_CLOSE(poly(x), cg(x), eps);
                CHECK_CLOSE(deriv(x), cg.derivative(x), eps);
            }
        }
    }

    TEST(CubicHermiteInterpolatorEG_5)
    {
        const double eps = 1.0e-10;
        const double xmin = 1.0;
        const double xmax = 2.0;
        const unsigned nPoints = 5;

        for (unsigned icycle=0; icycle<100; ++icycle)
        {
            const double xpeak = xmin + (xmax - xmin)*test_rng();
            const double ypeak = test_rng();
            const double a = (test_rng() - 0.5)*2.0;

            const Quadratic q(a, xpeak, ypeak);
            const QuadraticDerivative qd(a, xpeak);
            const CubicHermiteInterpolatorEG cg(xmin, xmax, nPoints, q, qd);

            if (a > 0.0)
            {
                double expectedX, expectedY;
                if (q(xmin) > q(xmax))
                {
                    expectedX = xmin;
                    expectedY = q(xmin);
                }
                else
                {
                    expectedX = xmax;
                    expectedY = q(xmax);
                }
                CHECK_CLOSE(expectedX, cg.argmax(), eps);
                CHECK_CLOSE(expectedY, cg.maximum(), eps);
                CHECK_CLOSE(xpeak, cg.location(), eps);
                CHECK_CLOSE(ypeak, cg(xpeak), eps);
            }
            else if (a < 0.0)
            {
                CHECK_CLOSE(xpeak, cg.argmax(), eps);
                CHECK_CLOSE(xpeak, cg.location(), eps);
                CHECK_CLOSE(ypeak, cg.maximum(), eps);
                CHECK_CLOSE(ypeak, cg(xpeak), eps);

                const double xplus = (xmax - xpeak)/2.0;
                const double yplus = cg(xpeak + xplus);
                const double deltaPlus = cg.maximum() - yplus;
                const double plusSig = cg.sigmaPlus(deltaPlus);
                CHECK_CLOSE(xplus, plusSig, eps);

                const double xminus = (xpeak - xmin)/3.0;
                const double yminus = cg(xpeak - xminus);
                const double deltaMinus = cg.maximum() - yminus;
                const double minusSig = cg.sigmaMinus(deltaMinus);
                CHECK_CLOSE(xminus, minusSig, eps);
            }
        }
    }

    TEST(CubicHermiteInterpolatorEG_6)
    {
        const double eps = 1.0e-12;
        const double xmin = 1.0;
        const double xmax = 2.0;
        const unsigned nPoints = 5;
        
        double coeffs1[4], coeffs2[4];
        for (unsigned icycle=0; icycle<10; ++icycle)
        {
            for (unsigned i=0; i<4; ++i)
            {
                coeffs1[i] = test_rng();
                coeffs2[i] = test_rng();
            }
            const double a = test_rng() - 0.5;
            const double b = test_rng() - 0.5;

            const Poly1D poly1(coeffs1, 3);
            const Poly1D& deriv1 = poly1.derivative();
            const Poly1D poly2(coeffs2, 3);
            const Poly1D& deriv2 = poly2.derivative();

            for (unsigned i=0; i<4; ++i)
                coeffs1[i] = a*coeffs1[i] + b*coeffs2[i];

            const Poly1D polySum(coeffs1, 3);
            const Poly1D& derivSum = polySum.derivative();
            const Poly1D& secDerivSum = derivSum.derivative();

            const CubicHermiteInterpolatorEG cg1(xmin, xmax, nPoints, poly1, deriv1);
            const CubicHermiteInterpolatorEG cg2(xmin, xmax, nPoints, poly2, deriv2);
            const auto& sum = a*cg1 + b*cg2;
            CHECK_EQUAL(xmin, sum.parMin());
            CHECK_EQUAL(xmax, sum.parMax());
            const unsigned sumN = std::round(1.0 + (sum.parMax() - sum.parMin())/sum.stepSize());
            CHECK_EQUAL(nPoints, sumN);

            for (unsigned itest=0; itest<10; ++itest)
            {
                const double x = get_x(xmin, xmax, itest);
                CHECK_CLOSE(polySum(x), sum(x), eps);
                CHECK_CLOSE(derivSum(x), sum.derivative(x), eps);
                CHECK_CLOSE(secDerivSum(x), sum.secondDerivative(x), eps);
            }
        }
    }
}
