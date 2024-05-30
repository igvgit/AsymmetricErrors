#ifndef ASE_TRANSITIONCUBIC_HH_
#define ASE_TRANSITIONCUBIC_HH_

#include <utility>
#include <cassert>
#include <stdexcept>
#include <algorithm>

#include "ase/mathUtils.hh"
#include "ase/Interval.hh"

namespace ase {
    // This class defines the cubic curve from a given value,
    // derivative, and second derivative at some point x0,
    // and the condition that the second derivative becomes
    // zero at x0 + h. Note that h must not be zero.
    template<typename Real>
    class TransitionCubic
    {
    public:
        typedef Real value_type;

        inline TransitionCubic(const Real i_x0, const Real i_h,
                               const Real valueAtX0,
                               const Real derivativeAtX0,
                               const Real secondDerivativeAtX0)
            : x0_(i_x0), h_(i_h), f0_(valueAtX0),
              der0_(derivativeAtX0), sder0_(secondDerivativeAtX0)
        {
            if (!h_) throw std::invalid_argument(
                "In ase::TransitionCubic constructor: "
                "argument h must be non-zero");
        }

        inline Real x0() const {return x0_;}
        inline Real h() const {return h_;}

        inline Real operator()(const Real x_in) const
        {
            const Real x = x_in - x0_;
            return ((sder0_/2 - sder0_/(6*h_)*x)*x + der0_)*x + f0_;
        }

        inline Real derivative(const Real x_in) const
        {
            const Real x = x_in - x0_;
            return der0_ + x*(sder0_ - (sder0_*x)/(2*h_));
        }

        inline Real secondDerivative(const Real x_in) const
        {
            const Real x = x_in - x0_;
            return sder0_*(1.0L - x/h_);
        }

        inline bool hasExtremum() const
        {
            return der0_*(der0_ + (h_*sder0_)/2) <= Real();
        }

        inline std::pair<Real,Real> extremum() const
        {
            if (hasExtremum())
            {
                if (der0_ == Real())
                    return std::pair<Real,Real>(x0_, (*this)(x0_));
                if (der0_ + (h_*sder0_)/2 == Real())
                    return std::pair<Real,Real>(x0_+h_, (*this)(x0_+h_));
                Real roots[2];
                const Real a = -sder0_/2/h_;
                const unsigned nRoots = solveQuadratic(
                    sder0_/a, der0_/a, &roots[0], &roots[1]);
                assert(nRoots == 2U);
                Interval<Real> support(0.0, h_, CLOSED_INTERVAL);
                unsigned nIn = 0U, lastIn = 2U;
                for (unsigned i=0; i<2U; ++i)
                    if (support.contains(roots[i]))
                    {
                        ++nIn;
                        lastIn = i;
                    }
                assert(nIn < 2U);
                if (nIn)
                    return std::pair<Real,Real>(roots[lastIn]+x0_, (*this)(roots[lastIn]+x0_));
                Real distance[2];
                for (unsigned i=0; i<2U; ++i)
                    distance[i] = support.distance(roots[i]);
                const unsigned ibest = distance[0] < distance[1] ? 0 : 1;
                const Real xmin = support.min();
                if (roots[ibest] < xmin)
                    return std::pair<Real,Real>(xmin+x0_, (*this)(xmin+x0_));
                else
                {
                    const Real xmax = support.max();
                    return std::pair<Real,Real>(xmax+x0_, (*this)(xmax+x0_));
                }
            }
            else
                return std::pair<Real,Real>(0.0, 0.0);
        }

    private:
        Real x0_;
        Real h_;
        Real f0_;
        Real der0_;
        Real sder0_;
    };
}

#endif // ASE_TRANSITIONCUBIC_HH_
