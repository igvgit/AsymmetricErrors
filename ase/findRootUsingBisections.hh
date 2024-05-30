#ifndef ASE_FINDROOTUSINGBISECTIONS_HH_
#define ASE_FINDROOTUSINGBISECTIONS_HH_

#include <cmath>
#include <limits>
#include <cassert>
#include <stdexcept>

namespace ase {
    /**
    // Numerical equation solving for 1-d functions using interval division.
    //
    // Input arguments are as follows:
    //
    //   f       -- The functor making up the equation to solve: f(x) == rhs.
    //
    //   rhs     -- The "right hand side" of the equation.
    //
    //   x0, x1  -- The starting interval for the search.
    //
    //   tol     -- Tolerance parameter. Typically, the found solution
    //              will be within a factor of 1 +- tol of the real one.
    //
    //   root    -- Location where the root will be written. This could
    //              also be a discontinuity point of f(x) or a singularity.
    //
    // The function returns "false" in case the initial interval does not
    // bracket the root. In this case *root is not modified.
    */
    template <class Functor, typename Real>
    bool findRootUsingBisections(const Functor& f, const Real rhs,
                                 Real x0, Real x1,
                                 const Real tol, Real* root)
    {
        if (tol <= std::numeric_limits<Real>::epsilon())
            throw std::invalid_argument("In ase::findRootUsingBisections: "
                                        "tolerance argument is too small");
        assert(root);

        const Real f0 = f(x0);
        if (f0 == rhs)
        {
            *root = x0;
            return true;
        }

        const Real f1 = f(x1);
        if (f1 == rhs)
        {
            *root = x1;
            return true;
        }

        const bool increasing = f0 < rhs && rhs < f1;
        const bool decreasing = f0 > rhs && rhs > f1;
        if (!(increasing || decreasing))
            return false;

        const Real sqrtol = std::sqrt(tol);
        const unsigned maxiter = 2000;
        for (unsigned iter=0; iter<maxiter; ++iter)
        {
            const Real xmid = (x0 + x1)/2.0;
            const Real fmid = f(xmid);

            if (fmid == rhs)
            {
                *root = xmid;
                return true;
            }
            if (std::abs(x0 - x1)/(std::abs(xmid) + sqrtol) <= tol)
            {
                *root = xmid;
                return true;
            }

            if (increasing)
            {
                if (fmid > rhs)
                    x1 = xmid;
                else
                    x0 = xmid;
            }
            else
            {
                if (fmid > rhs)
                    x0 = xmid;
                else
                    x1 = xmid;
            }
        }

        throw std::runtime_error("In ase::findRootUsingBisections: "
                                 "iterations faled to converge");
        return false;
    }
}

#endif // ASE_FINDROOTUSINGBISECTIONS_HH_
