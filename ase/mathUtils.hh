#ifndef ASE_MATHUTILS_HH_
#define ASE_MATHUTILS_HH_

#include <cmath>
#include <cassert>

#include "ase/HermiteProbOrthoPoly.hh"

namespace ase {
    /**
    // Simple factorial. Will generate a run-time error if n!
    // is larger than the largest unsigned long.
    */
    unsigned long factorial(unsigned n);

    /**
    // Factorial as a long double. Although imprecise, this has much
    // larger dynamic range than "factorial".
    */
    long double ldfactorial(unsigned n);

    /** Natural log of a factorial (using Stirling's series for large n) */
    long double logfactorial(unsigned long n);

    //@{
    /**
    // Solve the quadratic equation x*x + b*x + c == 0
    // in a numerically sound manner. Return the number of roots.
    */
    unsigned solveQuadratic(double b, double c,
                            double *x1, double *x2);
    unsigned solveQuadratic(long double b, long double c,
                            long double *x1, long double *x2);
    //@}

    /**
    // Find the real roots of the cubic:
    //   x**3 + p*x**2 + q*x + r = 0
    // The number of real roots is returned, and the roots are placed
    // into the "v3" array. Original code is by Don Herbison-Evans (see
    // his article "Solving Quartics and Cubics for Graphics" in the
    // book "Graphics Gems V", page 3), with minimal adaptation for
    // this package by igv.
    */
    unsigned solveCubic(double p, double q, double r, double v3[3]);

    /**
    // The location and the minimum value of a cubic polynomial function
    // on the interval [0, 1]. The cubic is specified by its values and
    // its derivatives at 0 and 1. The first element of the pair is
    // the location of the minimum (will be on [0, 1]) and the second is
    // the minimum value.
    */
    std::pair<double,double> cubicMinimum01(double f0, double d0,
                                            double f1, double d1);

    /**
    // The location and the maximum value of a cubic polynomial function
    // on the interval [0, 1]. The cubic is specified by its values and
    // its derivatives at 0 and 1. The first element of the pair is
    // the location of the maximum (will be on [0, 1]) and the second is
    // the maximum value.
    */
    std::pair<double,double> cubicMaximum01(double f0, double d0,
                                            double f1, double d1);

    /**
    // Sum of polynomial series. The length of the
    // array of coefficients should be at least degree+1.
    // The highest degree coefficient is assumed to be
    // the last one in the "coeffs" array (0th degree
    // coefficient comes first).
    */
    template<typename Numeric>
    inline long double polySeriesSum(const Numeric* coeffs,
                                     const unsigned degree,
                                     const long double x)
    {
        assert(coeffs);
        long double sum = 0.0L;
        for (int deg=degree; deg>=0; --deg)
        {
            sum *= x;
            sum += coeffs[deg];
        }
        return sum;
    }

    /**
    // Sum and derivative of polynomial series. The length of the
    // array of coefficients should be at least degree+1.
    */
    template<typename Numeric>
    inline void polyAndDeriv(const Numeric* coeffs, unsigned degree,
                             const long double x,
                             long double *value, long double *deriv)
    {
        assert(coeffs);
        long double sum = 0.0L, der = 0.0L;
        for (; degree>=1; --degree)
        {
            sum *= x;
            der *= x;
            sum += coeffs[degree];
            der += degree*coeffs[degree];
        }
        if (value)
            *value = sum*x + coeffs[0];
        if (deriv)
            *deriv = der;
    }

    /**
    // Series for Hermite polynomials orthogonal with weight exp(-x*x/2).
    // These are sometimes called the "probabilists' Hermite polynomials".
    */
    template<typename Numeric>
    inline long double hermiteSeriesSumProb(const Numeric* coeffs,
                                            const unsigned degree,
                                            const long double x)
    {
        assert(coeffs);
        long double result = coeffs[0], pminus2 = 1.0L, pminus1 = x;
        if (degree)
        {
            result += coeffs[1]*x;
            for (unsigned i=2; i<=degree; ++i)
            {
                const long double p = x*pminus1 - (i-1U)*pminus2;
                result += p*coeffs[i];
                pminus2 = pminus1;
                pminus1 = p;
            }
        }
        return result;
    }

    /** Series for Hermite polynomials together with its derivative */
    template<typename Numeric>
    inline void hermiteSeriesWithDeriv(const Numeric* coeffs,
                                       const unsigned degree,
                                       const long double x,
                                       long double* sum,
                                       long double* derivative)
    {
        assert(coeffs);
        assert(sum);
        assert(derivative);

        *sum = coeffs[0];
        *derivative = 0.0L;

        long double pminus2 = 1.0L, pminus1 = x;
        if (degree)
        {
            *sum += coeffs[1]*x;
            *derivative += coeffs[1];
            for (unsigned i=2; i<=degree; ++i)
            {
                const long double p = x*pminus1 - (i-1U)*pminus2;
                *sum += p*coeffs[i];
                *derivative += pminus1*i*coeffs[i];
                pminus2 = pminus1;
                pminus1 = p;
            }
        }
    }

    /** Series for Hermite polynomials together with its two derivatives */
    template<typename Numeric>
    inline void hermiteSeriesWithDeriv(const Numeric* coeffs,
                                       const unsigned degree,
                                       const long double x,
                                       long double* sum,
                                       long double* derivative,
                                       long double* secondDerivative)
    {
        assert(coeffs);
        assert(sum);
        assert(derivative);
        assert(secondDerivative);

        *sum = coeffs[0];
        *derivative = 0.0L;
        *secondDerivative = 0.0L;

        long double pminus2 = 1.0L, pminus1 = x;
        if (degree)
        {
            *sum += coeffs[1]*x;
            *derivative += coeffs[1];
            for (unsigned i=2; i<=degree; ++i)
            {
                const long double p = x*pminus1 - (i-1U)*pminus2;
                *sum += p*coeffs[i];
                *derivative += pminus1*i*coeffs[i];
                *secondDerivative += pminus2*i*(i-1U)*coeffs[i];
                pminus2 = pminus1;
                pminus1 = p;
            }
        }
    }

    /**
    // If you have series for Hermite polynomials orthogonal with weight
    // exp(-x*x/2), this function will calculate the series expansion
    // coefficients for the derivative. The length of the array of
    // coefficients should be at least degree+1, and the length of the
    // "result" array should be at least degree.
    */
    template<typename Numeric>
    inline void hermiteSeriesDerivative(const Numeric* coeffs,
                                        const unsigned degree,
                                        Numeric *result)
    {
        assert(coeffs);
        if (degree)
        {
            assert(result);
            for (unsigned deg=0; deg<degree; ++deg)
                result[deg] = coeffs[deg+1U]*static_cast<Numeric>(deg+1U);
        }
    }

    /** 
    // Derive the coefficients of the Hermite polynomial series.
    // The length of the array "coeffs" should be at least maxdeg + 1.
    // Note that the coefficients are returned in the order of increasing
    // degree (same order is used by various "hermiteSeries" functions).
    // The code assumes that the function can be represented by the
    // polynomial of degree maxdeg exactly.
    */
    template <class Functor>
    inline void hermiteSeriesCoeffs(const Functor& fcn, long double* coeffs,
                                    const unsigned maxdeg)
    {
        const HermiteProbOrthoPoly poly;
        poly.calculateCoeffs(fcn, coeffs, maxdeg);
        long double factorialSqrt = 1.0L;
        for (unsigned i=2; i<=maxdeg; ++i)
        {
            factorialSqrt *= HermiteProbOrthoPoly::fast_sqrtl(i);
            coeffs[i] /= factorialSqrt;
        }
    }

    /**
    // Real roots of Hermite polynomial series on the interval (a, b).
    // The algorithm implemented here is rather trivial and should not
    // be used for high degree polynomials. The number of roots is
    // returned. Multiple roots are counted only once. The length of
    // the "roots" array should be at least "degree".
    */
    unsigned hermiteSeriesRoots(const long double* coeffs,
                                unsigned degree,
                                long double a, long double b,
                                long double* roots);

    /**
    // Series for the Chebyshev polynomials of the first kind. The array
    // of coefficients must be at least degree+1 long.
    */
    template<typename Numeric>
    inline long double chebyshevSeriesSum(const Numeric *coeffs,
                                          const unsigned degree,
                                          const long double x)
    {
        assert(coeffs);
        const long double twox = 2.0L*x;

        // Clenshaw recursion
        long double rp2 = 0.0L, rp1 = 0.0L, r = 0.0L;
        for (unsigned k = degree; k > 0U; --k)
        {
            r = twox*rp1 - rp2 + coeffs[k];
            rp2 = rp1;
            rp1 = r;
        }
        return x*rp1 - rp2 + coeffs[0];
    }

    /**
    // Series for the Chebyshev polynomials of the first kind
    // on the given interval [xmin, xmax]
    */
    template<typename Numeric>
    inline long double chebyshevSeriesSum(const Numeric *coeffs,
                                          const unsigned degree,
                                          const long double xmin,
                                          const long double xmax,
                                          const long double x)
    {
        assert(xmin != xmax);
        const long double xtrans = 2.0L*(x - xmin)/(xmax - xmin) - 1.0L;
        return chebyshevSeriesSum(coeffs, degree, xtrans);
    }

    double linearValue(double x0, double y0, double x1, double y1, double x);
}

#endif // ASE_MATHUTILS_HH_
