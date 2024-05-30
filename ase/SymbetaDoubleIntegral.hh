#ifndef ASE_SYMBETADOUBLEINTEGRAL_HH_
#define ASE_SYMBETADOUBLEINTEGRAL_HH_

#include <cmath>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <memory>
#include <utility>

#include "ase/Poly1D.hh"
#include "ase/mathUtils.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/TDeriv.hh"
#include "ase/Interval.hh"

namespace ase {
    template<typename Real>
    class SymbetaDoubleIntegral
    {
    public:
        typedef Real value_type;

        // Default constructor creates an identity transform
        inline SymbetaDoubleIntegral()
            : h_(1.0), a_(0.0), k_(1.0), p_(0)
        {
            initialize();
        }

        inline SymbetaDoubleIntegral(const unsigned i_p, const Real i_h,
                                     const Real i_a, const Real i_k)
            : h_(i_h), a_(i_a), k_(i_k), p_(i_p)
        {
            initialize();
        }

        inline unsigned p() const {return p_;}
        inline Real h() const {return h_;}
        inline Real a() const {return a_;}
        inline Real k() const {return k_;}

        inline bool hasExtremum() const
            {return derivative(-h_)*derivative(h_) <= 0.0;}

        inline bool isFlat() const
            {return derivative(-h_)*derivative(h_) == 0.0;}

        // If extremum exists, the first element of the returned pair
        // will be its location and the second the function value there.
        // If the extremum does not exist, the result is undefined.
        // Call the "hasExtremum()" function to see if the output of
        // this method is meaningful.
        inline std::pair<Real,Real> extremum() const
            {return std::pair<Real,Real>(extremumArg_, extremumValue_);}

        inline Real operator()(const Real x) const
        {
            if (x <= -h_)
                return a_*(leftDer_*(x + h_) + leftValue_) + k_*x;
            else if (x >= h_)
                return a_*(rightDer_*(x - h_) + rightValue_) + k_*x;
            else
                return a_*h_*h_*poly_(x/h_) + k_*x;
        }

        inline Real derivative(const Real x) const
        {
            if (x <= -h_)
                return a_*leftDer_ + k_;
            else if (x >= h_)
                return a_*rightDer_ + k_;
            else
                return a_*h_*dpoly_(x/h_) + k_;
        }

        inline Real secondDerivative(const Real x) const
        {
            if (std::abs(x) < h_)
            {
                const Real r = x/h_;
                return a_*std::pow(1.0 - r*r, p_);
            }
            else
                return 0.0;
        }

        inline Real zoneContinuation(const Real xZone, const Real x) const
        {
            if (xZone <= -h_)
                return a_*(leftDer_*(x + h_) + leftValue_) + k_*x;
            else if (xZone >= h_)
                return a_*(rightDer_*(x - h_) + rightValue_) + k_*x;
            else
                return a_*h_*h_*poly_(x/h_) + k_*x;
        }

        // The inverse function can be single- or double-valued.
        // The return value of the following function is the number
        // of solutions found for the equation (*this)(x) == y.
        inline unsigned inverse(const Real y, Real solutions[2]) const
        {
            if (hasExtremum())
                return inverseWithExtremum(y, solutions);
            else
                return monotonousInverse(y, solutions);
        }

        static inline SymbetaDoubleIntegral<Real> fromSigmas(
            const unsigned i_p, const Real i_h,
            const Real sigmaPlus, const Real sigmaMinus,
            const bool normalizeScale = false)
        {
            Real sp = sigmaPlus;
            Real sm = sigmaMinus;
            if (normalizeScale)
            {
                const Real s = (std::abs(sigmaPlus) +
                                std::abs(sigmaMinus))/2.0;
                sp /= s;
                sm /= s;
            }
            const SymbetaDoubleIntegral<Real> tmp(i_p, i_h, 1.0, 0.0);
            const Real v1 = tmp(1.0);
            const Real a = (sp - sm)/v1/2.0;
            const Real k = (sp + sm)/2.0;
            return SymbetaDoubleIntegral<Real>(i_p, i_h, a, k);
        }

        static inline Real minRNoExtremum(const unsigned i_p, const Real i_h)
        {
            const Real tol = 2.0*std::numeric_limits<Real>::epsilon();
            const Real sqrtol = std::sqrt(tol);
            const unsigned maxiter = 2000;
            Real x0 = 0.0;
            Real x1 = 1.0;
            Real r = (x0 + x1)/2.0;
            for (unsigned iter=0; iter<maxiter; ++iter)
            {
                const Real sp = 2.0*r/(1.0 + r);
                const Real sm = 2.0/(1.0 + r);
                const SymbetaDoubleIntegral<Real>& sdi = fromSigmas(
                    i_p, i_h, sp, sm);
                if (sdi.hasExtremum())
                    x0 = r;
                else
                    x1 = r;
                r = (x0 + x1)/2.0;
                if ((x1 - x0)/(r + sqrtol) <= tol)
                    return x1;
            }
            throw std::runtime_error(
                "In ase::SymbetaDoubleIntegral::minRNoExtremum: "
                "algorithm failed to converge");
            return 0.0;
        }

    private:
        inline void initialize()
        {
            // This method initializes all internal
            // variables from h_, a_, k_, and p_
            if (h_ <= 0.0) throw std::invalid_argument(
                "In ase::SymbetaDoubleIntegral::initialize: "
                "parameter h must be positive");
            if (p_ > 20U) throw std::invalid_argument(
                "In ase::SymbetaDoubleIntegral::initialize: "
                "parameter p is too large");
            std::vector<long double> coeffs(2U*p_ + 1U, 0.0L);
            const unsigned long pfact = factorial(p_);
            for (unsigned i=0; i<=p_; ++i)
            {
                const unsigned deg = 2U*i;
                coeffs[deg] = pfact/factorial(i)/factorial(p_ - i);
                if (i % 2U)
                    coeffs[deg] *= -1.0L;
            }
            const Poly1D ddpoly(&coeffs[0], 2U*p_);
            dpoly_ = ddpoly.integral(0.0L);
            poly_ = dpoly_.integral(0.0L);
            const Real hsq = h_*h_;
            leftDer_ = h_*dpoly_(-1.0);
            leftValue_ = hsq*poly_(-1.0);
            rightDer_ = h_*dpoly_(1.0);
            rightValue_ = hsq*poly_(1.0);
            vAtMH_ = (*this)(-h_);
            vAtH_ = (*this)(h_);

            extremumArg_ = determineExtremumLocation();
            extremumValue_ = (*this)(extremumArg_);
        }

        inline Real determineExtremumLocation() const
        {
            static const Real tol = 2*std::numeric_limits<Real>::epsilon();

            if (hasExtremum() && k_ != 0.0)
            {
                if (derivative(-h_) == 0.0)
                    return -h_;
                if (derivative(h_) == 0.0)
                    return h_;
                Real tmp;
                const Real zero = 0.0;
                const bool status = findRootUsingBisections(
                    Private::TDeriv<SymbetaDoubleIntegral<Real> >(*this),
                    zero, -h_, h_, tol, &tmp);
                assert(status);
                return tmp;
            }
            return 0.0;
        }

        inline unsigned monotonousInverse(const Real y, Real solutions[2]) const
        {
            static const Real tol = 2.0*std::numeric_limits<Real>::epsilon();

            const bool increasing = derivative(0) > 0.0;
            if ((increasing && y >= vAtH_) || (!increasing && y <= vAtH_))
            {
                solutions[0] = (y - vAtH_)/(a_*rightDer_ + k_) + h_;
                return 1U;
            }
            if ((increasing && y <= vAtMH_) || (!increasing && y >= vAtMH_))
            {
                solutions[0] = (y - vAtMH_)/(a_*leftDer_ + k_) - h_;
                return 1U;
            }

            const Interval<Real> yrange(vAtMH_, vAtH_, OPEN_INTERVAL);
            assert(yrange.contains(y));
            const bool status = findRootUsingBisections(
                *this, y, -h_, h_, tol, &solutions[0]);
            assert(status);
            return 1U;
        }

        inline unsigned inverseWithExtremum(const Real y, Real solutions[2]) const
        {
            static const Real tol = 2.0*std::numeric_limits<Real>::epsilon();

            if (y == extremumValue_)
            {
                solutions[0] = extremumArg_;
                solutions[1] = extremumArg_;
                return 2U;
            }

            const bool isMaximum = secondDerivative(extremumArg_) < 0.0;
            if ((isMaximum && y > extremumValue_) || (!isMaximum && y < extremumValue_))
                return 0U;

            // Find the solution to the left of the extremum
            if ((isMaximum && y <= vAtMH_) || (!isMaximum && y >= vAtMH_))
                solutions[0] = (y - vAtMH_)/(a_*leftDer_ + k_) - h_;
            else
            {
                const Interval<Real> yrange(extremumValue_, vAtMH_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, -h_, extremumArg_, tol, &solutions[0]);
                assert(status);
            }

            // Find the solution to the right of the extremum
            if ((isMaximum && y <= vAtH_) || (!isMaximum && y >= vAtH_))
                solutions[1] = (y - vAtH_)/(a_*rightDer_ + k_) + h_;
            else
            {
                const Interval<Real> yrange(extremumValue_, vAtH_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, extremumArg_, h_, tol, &solutions[1]);
                assert(status);
            }

            return 2U;
        }

        Real h_;
        Real a_;
        Real k_;
        unsigned p_;

        Poly1D poly_;
        Poly1D dpoly_;

        Real leftDer_;
        Real leftValue_;
        Real rightDer_;
        Real rightValue_;
        Real vAtMH_;
        Real vAtH_;

        Real extremumArg_;
        Real extremumValue_;
    };

    template<typename Real>
    class SDIZoneFunctor
    {
    public:
        inline SDIZoneFunctor(const SymbetaDoubleIntegral<Real>& curve,
                              const Real xZone, const Real fShift,
                              const unsigned power)
            : curve_(curve), xZone_(xZone), fShift_(fShift), power_(power) {}

        inline Real operator()(const Real x) const
        {
            if (power_)
            {
                const Real del = curve_.zoneContinuation(xZone_, x) - fShift_;
                switch (power_)
                {
                case 1U:
                    return del;
                case 2U:
                    return del*del;
                case 3U:
                    return del*del*del;
                case 4U:
                    {
                        const Real delsq = del*del;
                        return delsq*delsq;
                    }
                default:
                    return std::pow(del, power_);
                }
            }
            else
                return 1.0;
        }

    private:
        const SymbetaDoubleIntegral<Real>& curve_;
        Real xZone_;
        Real fShift_;
        unsigned power_;
    };
}

#endif // ASE_SYMBETADOUBLEINTEGRAL_HH_
