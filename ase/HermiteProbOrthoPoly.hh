#ifndef ASE_HERMITEPROBORTHOPOLY_HH_
#define ASE_HERMITEPROBORTHOPOLY_HH_

#include <cassert>
#include <utility>

#include "ase/GaussHermiteQuadrature.hh"

namespace ase {
    namespace Private {
        template <class SomePoly, class Functor>
        class PolyFunctorProd
        {
        public:
            inline PolyFunctorProd(const SomePoly& p,
                                   const Functor& f, const unsigned d1)
                : poly(p), fcn(f), deg(d1) {}

            inline long double operator()(const long double x) const
                {return fcn(x)*poly.poly(deg, x);}

        private:
            const SomePoly& poly;
            const Functor& fcn;
            unsigned deg;
        };
    }

    /** Orthonormal(!) probabilist's Hermite polynomials */
    class HermiteProbOrthoPoly
    {
    public:
        /** Polynomial values */
        long double poly(unsigned deg, long double x) const;

        /**
        // Values of all orthonormal polynomials up to some degree.
        // Faster than calling "poly" multiple times. The size of
        // the "values" array should be at least maxdeg + 1.
        */
        void allpoly(long double x, long double* values, unsigned maxdeg) const;

        /** Weight function with which the polynomials are orthonormal */
        long double weight(long double x) const;

        /** Integral of the weight function from -Infinity to x */
        long double weightIntegral(long double x) const;

        /**
        // Polynomial series. The size of the "coeffs" array should
        // be at least maxdeg + 1.
        */
        long double series(const long double* coeffs,
                           unsigned maxdeg, long double x) const;

        /**
        // Build the coefficients of the orthogonal polynomial series
        // for the given function. The length of the array "coeffs"
        // should be at least maxdeg + 1. Note that the coefficients
        // are returned in the order of increasing degree (same order
        // is used by the "series" function). The code assumes that
        // the function can be represented by the polynomial of
        // degree maxdeg exactly.
        */
        template <class Functor>
        inline void calculateCoeffs(const Functor& fcn, long double* coeffs,
                                    const unsigned maxdeg) const
        {
            assert(coeffs);
            const unsigned nPt = GaussHermiteQuadrature::minimalExactRule(2U*maxdeg);
            const GaussHermiteQuadrature ghq(nPt);

            for (unsigned i=0; i<=maxdeg; ++i)
            {
                const Private::PolyFunctorProd<HermiteProbOrthoPoly,Functor>
                    prod(*this, fcn, i);
                coeffs[i] = ghq.integrateProb(0.0L, 1.0L, prod);
            }
        }

        /**
        // Integral from -Infinity to x of f(t)*Exp[-t^2/2]/Sqrt[2 Pi],
        // where f(t) is represented by its series expansion
        */
        long double weightedSeriesIntegral(const long double* coeffs,
                                           unsigned maxdeg, long double x) const;

        /** Recurrence coefficients */
        std::pair<long double,long double>
        recurrenceCoeffs(unsigned k) const;

        /** 
        // If needed, fast square root of unsigned arguments
        // (has a built-in precomputed table for 100 numbers or so)
        */
        static long double fast_sqrtl(unsigned u);
    };
}

#endif // ASE_HERMITEPROBORTHOPOLY_HH_
