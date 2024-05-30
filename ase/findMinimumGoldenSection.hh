#ifndef ASE_FINDMINIMUMGOLDENSECTION_HH_
#define ASE_FINDMINIMUMGOLDENSECTION_HH_

#include <cmath>
#include <cassert>
#include <stdexcept>

namespace ase {
    template <typename Real>
    inline bool isMinimumBracketed(const Real fleft,
                                   const Real fmiddle,
                                   const Real fright)
    {
        if (!(fmiddle <= fleft && fmiddle <= fright))
            return false;
        if (fmiddle == fleft && fmiddle == fright)
            return false;
        return true;
    }

    /**
    // This function searches for a minimum of a functor using the
    // golden section method. Arguments xleft, xmiddle, and xright
    // should bracket the minimum. "tol" is the tolerance parameter.
    // Note that, making the tolerance much smaller than
    // std::sqrt(std::numeric_limits<Real>::epsilon()) is typically not
    // going to improve the precision of the minumum location (returned
    // in *argmin).
    //
    // The function returns "true" if it finds the minimum, "false" otherwise.
    */
    template <class Functor, typename Real>
    bool findMinimumGoldenSection(const Functor& f,
                                  Real xleft, Real xmiddle, Real xright,
                                  const Real tol, Real* argmin,
                                  Real* fmin = 0)
    {
        static const Real zero = Real();
        static const Real frac = (3.0 - std::sqrt(static_cast<Real>(5)))/2.0;
        static const unsigned maxIter = 2000;

        assert(argmin);
        assert(xleft < xmiddle);
        assert(xmiddle < xright);
        assert(tol > zero);

        const Real sqrtol = std::sqrt(tol);
        Real fleft = f(xleft);
        Real fmiddle = f(xmiddle);
        Real fright = f(xright);
        if (!isMinimumBracketed(fleft, fmiddle, fright))
            return false;

        for (unsigned iter = 0; iter < maxIter; ++iter)
        {
            const Real lenRight = xright - xmiddle;
            assert(lenRight > zero);
            const Real lenLeft = xmiddle - xleft;
            assert(lenLeft > zero);
            const bool splitRight = lenRight > lenLeft;
            const Real xnext = splitRight ? xmiddle + lenRight*frac :
                                            xmiddle - lenLeft*frac;
            const Real fnext = f(xnext);
            if (splitRight)
            {
                if (fnext < fmiddle || (fnext == fmiddle && fright > fmiddle))
                {
                    xleft = xmiddle;
                    fleft = fmiddle;
                    xmiddle = xnext;
                    fmiddle = fnext;
                }
                else
                {
                    xright = xnext;
                    fright = fnext;
                }
            }
            else
            {
                if (fnext < fmiddle || (fnext == fmiddle && fleft > fmiddle))
                {
                    xright = xmiddle;
                    fright = fmiddle;
                    xmiddle = xnext;
                    fmiddle = fnext;
                }
                else
                {
                    xleft = xnext;
                    fleft = fnext;
                }
            }
            assert (!(fmiddle == fleft && fmiddle == fright));

            if ((xright - xleft)/(std::abs(xmiddle) + sqrtol) < tol)
            {
                *argmin = xmiddle;
                if (fmin)
                    *fmin = fmiddle;
                return true;
            }
        }

        throw std::runtime_error("In ase::findMinimumGoldenSection: "
                                 "iterations faled to converge");
        return false;
    }
}

#endif // ASE_FINDMINIMUMGOLDENSECTION_HH_
