#include <algorithm>

#include "ase/LegendreOrthoPoly1D.hh"
#include "ase/mathUtils.hh"

// Calculate (-1)^k
static inline int powm1(const unsigned k)
{
    return k % 2U ? -1 : 1;
}

namespace ase {
    void LegendreOrthoPoly1D::monomialCoeffs(
        const double* coeffs, const unsigned degree,
        long double* monoCoeffs) const
    {
        assert(coeffs);
        assert(monoCoeffs);

        std::fill(monoCoeffs, monoCoeffs+(degree+1U), 0.0L);

        for (unsigned deg=0U; deg<=degree; ++deg)
            if (coeffs[deg])
            {
                const long double norm = coeffs[deg]*sqrtl(2.0L*deg + 1)/powl(2.0L, deg);
                const unsigned kmax = deg/2U;
                for (unsigned k=0; k<=kmax; ++k)
                    monoCoeffs[deg-2U*k] += norm*powm1(k)*ldfactorial(2U*(deg-k))/ldfactorial(k)/ldfactorial(deg-k)/ldfactorial(deg-2U*k);
            }
    }

    void LegendreOrthoPoly1D::integralCoeffs(
        const double* coeffs, const unsigned degree,
        double* integCoeffs) const
    {
        const double sqr3 = 1.7320508075688773;

        assert(coeffs);
        assert(integCoeffs);

        integCoeffs[0] = coeffs[0];
        integCoeffs[1] = coeffs[0]/sqr3;
        std::fill(integCoeffs+2U, integCoeffs+(degree+2U), 0.0);

        for (unsigned deg=1U; deg<=degree; ++deg)
            if (coeffs[deg])
            {
                const double tmp = coeffs[deg]/sqrt(2.0*deg + 1);
                integCoeffs[deg+1U] += tmp/sqrt(2.0*deg + 3);
                integCoeffs[deg-1U] -= tmp/sqrt(2.0*deg - 1);
            }
    }

    void LegendreOrthoPoly1D::derivativeCoeffs(
        const double* coeffs, const unsigned degree,
        double* derivCoeffs) const
    {
        if (degree)
        {
            assert(coeffs);
            assert(derivCoeffs);

            std::fill(derivCoeffs, derivCoeffs+degree, 0.0);

            for (unsigned deg=1U; deg<=degree; ++deg)
                if (coeffs[deg])
                {
                    const double tmp = coeffs[deg]*sqrt(2.0*deg + 1);
                    for (unsigned k=0U; k<deg; k+=2U)
                        derivCoeffs[deg-1U-k] += tmp*sqrt(2.0*(deg-k) - 1);
                }
        }
        else
        {
            if (derivCoeffs)
                derivCoeffs[0] = 0.0;
        }
    }

    long double LegendreOrthoPoly1D::poly(
        const unsigned degree, const long double x) const
    {
        long double polyk = 1.0L;
        if (degree)
        {
            long double polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
            }
        }
        return polyk;
    }

    void LegendreOrthoPoly1D::allpoly(const long double x,
                                      long double* values,
                                      const unsigned degree) const
    {
        assert(values);
        values[0] = 1.0L;
        if (degree)
        {
            long double polyk = 1.0L, polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
                values[k+1] = p;
            }
        }
    }
}
