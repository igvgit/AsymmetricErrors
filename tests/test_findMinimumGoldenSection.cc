#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/findMinimumGoldenSection.hh"

using namespace ase;

namespace {
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

    TEST(findMinimumGoldenSection_1)
    {
        const double eps = 1.0e-14;

        const Quadratic q(1.0, 1.0, 3.0);
        double argmin, argval;
        bool status = findMinimumGoldenSection(
            q, -1.0, 0.0, 2.0, eps, &argmin, &argval);
        CHECK(status);
        CHECK_CLOSE(1.0, argmin, sqrt(eps));
        CHECK_CLOSE(3.0, argval, eps);

        status = findMinimumGoldenSection(
            q, 0.0, 2.0, 3.0, eps, &argmin, &argval);
        CHECK(status);
        CHECK_CLOSE(1.0, argmin, sqrt(eps));
        CHECK_CLOSE(3.0, argval, eps);
    }

    TEST(findMinimumGoldenSection_2)
    {
        const double eps = 1.0e-14;

        double argmin, argval;
        for (unsigned i=0; i<100; ++i)
        {
            const double a = test_rng() + 0.1;
            const double peakX = test_rng() - 0.5;
            const double peakY = test_rng() - 0.5;
            const Quadratic q(a, peakX, peakY);
            const double xmin = peakX - a*(test_rng() + 0.5);
            const double xmax = peakX + a*(test_rng() + 0.5);
            const double xmed = peakX + a*(test_rng() - 0.5);
            const bool status = findMinimumGoldenSection(
                q, xmin, xmed, xmax, eps, &argmin, &argval);            
            CHECK(status);
            CHECK_CLOSE(peakX, argmin, sqrt(eps));
            CHECK_CLOSE(peakY, argval, eps);
        }
    }
}
