#ifndef ASE_LEGENDREORTHOPOLY1D_HH_
#define ASE_LEGENDREORTHOPOLY1D_HH_

#include <cmath>
#include <cassert>
#include <utility>

namespace ase {
    /** Orthonormal Legendre polynomials on [-1, 1] */
    class LegendreOrthoPoly1D
    {
    public:
        inline ~LegendreOrthoPoly1D() {}

        inline long double weight(const long double x) const
            {return (x >= -1.0L && x <= 1.0L) ? 0.5L : 0.0L;}
        inline double xmin() const {return -1.0L;}
        inline double xmax() const {return 1.0L;}

        /**
        // Derive the series coefficients for the integral of the
        // argument series from -1 to x. The array of coefficients
        // must be at least degree+1 long, and the buffer for the
        // integral coefficients must be at least degree+2 long.
        */
        void integralCoeffs(const double* coeffs, unsigned degree,
                            double* integCoeffs) const;
        /**
        // Derive the series coefficients for the derivative of the
        // argument series. The array of coefficients must be at least
        // degree+1 long, and the buffer for the derivative coefficients
        // must be at least degree long.
        */
        void derivativeCoeffs(const double* coeffs, unsigned degree,
                              double* derivCoeffs) const;
        /**
        // Derive the series coefficients for the monomial
        // expansion. The arrays of coefficients must be
        // at least degree+1 long.
        */
        void monomialCoeffs(const double* coeffs, unsigned degree,
                            long double* monoCoeffs) const;

        /** Polynomial values */
        long double poly(unsigned deg, long double x) const;

        /**
        // All polynomial values up to the given degree. The length
        // of the "values" array must be at least degree + 1.
        */
        void allpoly(long double x, long double* values, unsigned degree) const;

        /** Polynomial series */
        template<class Real>
        inline Real series(const Real* coeffs, const unsigned degree,
                           const Real xIn) const
        {
            assert(coeffs);
            long double sum = coeffs[0];
            if (degree)
            {
                const long double x = xIn;
                long double polyk = 1.0L, polykm1 = 0.0L;
                std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
                for (unsigned k=0; k<degree; ++k)
                {
                    const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                    const long double p = ((x - rcurrent.first)*polyk -
                                           rcurrent.second*polykm1)/rnext.second;
                    sum += p*coeffs[k+1];
                    polykm1 = polyk;
                    polyk = p;
                    rcurrent = rnext;
                }
            }
            return sum;
        }

        /** Recurrence coefficients */
        inline std::pair<long double,long double>
        recurrenceCoeffs(const unsigned k) const
        {
            long double sqb = 1.0L;
            if (k)
                sqb = 1.0L/sqrtl(4.0L - 1.0L/k/k);
            return std::pair<long double,long double>(0.0L, sqb);
        }
    };
}

#endif // ASE_LEGENDREORTHOPOLY1D_HH_
