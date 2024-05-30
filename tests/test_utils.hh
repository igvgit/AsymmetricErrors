#ifndef TEST_UTILS_HH_
#define TEST_UTILS_HH_

#include <cassert>

double test_rng();

template <typename FCN>
double simpson_integral(const FCN& f,
                        const double xmin, const double xmax,
                        const unsigned nIntervals = 1000U)
{
    assert(nIntervals);
    const double step = (xmax - xmin)/nIntervals;
    long double sum = f(xmin);
    sum += 4.0*f(xmin + step/2.0);
    for (unsigned i=1; i<nIntervals; ++i)
    {
        sum += 2.0*f(xmin + i*step);
        sum += 4.0*f(xmin + (i+0.5)*step);
    }
    sum += f(xmax);
    return sum/6.0*step;
}

#endif // TEST_UTILS_HH_
