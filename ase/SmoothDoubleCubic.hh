#ifndef ASE_SMOOTHDOUBLECUBIC_HH_
#define ASE_SMOOTHDOUBLECUBIC_HH_

#include <cmath>
#include <cassert>
#include <limits>
#include <utility>
#include <stdexcept>

#include "ase/Interval.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/TDeriv.hh"

namespace ase {
    template<typename Real>
    class SmoothDoubleCubic
    {
    public:
        typedef Real value_type;

        inline SmoothDoubleCubic(const Real i_sigmaPlus,
                                 const Real i_sigmaMinus,
                                 const Real scaleNorm = 1.0)
            : sigmaPlus_(i_sigmaPlus/scaleNorm),
              sigmaMinus_(i_sigmaMinus/scaleNorm),
              leftDer_((5.0*sigmaMinus_ - sigmaPlus_)/4.0),
              rightDer_((5.0*sigmaPlus_ - sigmaMinus_)/4.0)
        {
            if (scaleNorm <= 0.0) throw std::invalid_argument(
                "In ase::SmoothDoubleCubic constructor: "
                "scale normalization factor must be positive");
            extremumArg_ = determineExtremumLocation();
            extremumValue_ = (*this)(extremumArg_);
        }

        inline Real sigmaPlus() const {return sigmaPlus_;}
        inline Real sigmaMinus() const {return sigmaMinus_;}
        inline bool hasExtremum() const {return leftDer_*rightDer_ <= 0.0;}
        inline bool isFlat() const {return leftDer_*rightDer_ == 0.0;}

        // If extremum exists, the first element of the returned pair
        // will be its location and the second the function value there.
        // If the extremum does not exist, the result is undefined.
        // Call the "hasExtremum()" function to see if the output of
        // this method is meaningful.
        inline std::pair<Real,Real> extremum() const
            {return std::pair<Real,Real>(extremumArg_, extremumValue_);}

        inline Real operator()(const Real x) const
        {
            if (x < -1.0)
                return leftDer_*(x + 1.0) - sigmaMinus_;
            else if (x < 0.0)
                return x*((sigmaPlus_ - sigmaMinus_)/4*(x*(x + 3) + 2) + sigmaMinus_);
            else if (x < 1.0)
                return x*((sigmaMinus_ - sigmaPlus_)/4*(x*(x - 3) + 2) + sigmaPlus_);
            else
                return rightDer_*(x - 1.0) + sigmaPlus_;
        }

        inline Real derivative(const Real x) const
        {
            if (x < -1.0)
                return leftDer_;
            else if (x < 0.0)
            {
                const Real sDel = sigmaPlus_ - sigmaMinus_;
                const Real a = 0.75*sDel;
                const Real b = 1.5*sDel;
                const Real c = (sigmaPlus_ + sigmaMinus_)/2.0;
                return x*(a*x  + b) + c;
            }
            else if (x < 1.0)
            {
                const Real sDel = sigmaMinus_ - sigmaPlus_;
                const Real a = 0.75*sDel;
                const Real b = -1.5*sDel;
                const Real c = (sigmaPlus_ + sigmaMinus_)/2.0;
                return x*(a*x  + b) + c;
            }
            else
                return rightDer_;
        }

        inline Real secondDerivative(const Real x) const
        {
            if (x < -1.0)
                return 0.0;
            else if (x < 0.0)
                return 3*(sigmaPlus_ - sigmaMinus_)*(1 + x)/2;
            else if (x < 1.0)
                return 3*(sigmaMinus_ - sigmaPlus_)*(x - 1)/2;
            else
                return 0.0;
        }

        inline Real zoneContinuation(const Real xZone, const Real x) const
        {
            if (xZone < -1.0)
                return leftDer_*(x + 1.0) - sigmaMinus_;
            else if (xZone < 0.0)
                return x*((sigmaPlus_ - sigmaMinus_)/4*(x*(x + 3) + 2) + sigmaMinus_);
            else if (xZone < 1.0)
                return x*((sigmaMinus_ - sigmaPlus_)/4*(x*(x - 3) + 2) + sigmaPlus_);
            else
                return rightDer_*(x - 1.0) + sigmaPlus_;
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

    private:
        inline unsigned monotonousInverse(const Real y, Real solutions[2]) const
        {
            static const Real tol = 2.0*std::numeric_limits<Real>::epsilon();

            const Real one = 1.0;
            const bool increasing = derivative(0) > 0.0;
            if ((increasing && y >= sigmaPlus_) || (!increasing && y <= sigmaPlus_))
            {
                solutions[0] = (y - sigmaPlus_)/rightDer_ + one;
                return 1U;
            }
            if ((increasing && y <= -sigmaMinus_) || (!increasing && y >= -sigmaMinus_))
            {
                solutions[0] = (y + sigmaMinus_)/leftDer_ - one;
                return 1U;
            }

            const Interval<Real> yrange(-sigmaMinus_, sigmaPlus_, OPEN_INTERVAL);
            assert(yrange.contains(y));
            const bool status = findRootUsingBisections(
                *this, y, -one, one, tol, &solutions[0]);
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
            const Real one = 1.0;
            if ((isMaximum && y <= -sigmaMinus_) || (!isMaximum && y >= -sigmaMinus_))
                solutions[0] = (y + sigmaMinus_)/leftDer_ - one;
            else
            {
                const Interval<Real> yrange(extremumValue_, -sigmaMinus_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, -one, extremumArg_, tol, &solutions[0]);
                assert(status);
            }

            // Find the solution to the right of the extremum
            if ((isMaximum && y <= sigmaPlus_) || (!isMaximum && y >= sigmaPlus_))
                solutions[1] = (y - sigmaPlus_)/rightDer_ + one;
            else
            {
                const Interval<Real> yrange(extremumValue_, sigmaPlus_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, extremumArg_, one, tol, &solutions[1]);
                assert(status);
            }

            return 2U;
        }

        inline Real determineExtremumLocation() const
        {
            if (hasExtremum())
            {
                static const Real tol = 2.0*std::numeric_limits<Real>::epsilon();

                const Real zero = 0.0;
                const Real minusOne = -1.0;
                const Real one = 1.0;
                if (sigmaPlus_ == -sigmaMinus_)
                    return zero;
                if (leftDer_ == zero)
                    return minusOne;
                if (rightDer_ == zero)
                    return one;
                Real tmp;
                const bool status = findRootUsingBisections(
                    Private::TDeriv<SmoothDoubleCubic<Real> >(*this),
                    zero, minusOne, one, tol, &tmp);
                assert(status);
                return tmp;
            }
            return 0.0;
        }

        Real sigmaPlus_;
        Real sigmaMinus_;
        Real leftDer_;
        Real rightDer_;
        Real extremumArg_;
        Real extremumValue_;
    };

    template<typename Real>
    class SDCZoneFunctor
    {
    public:
        inline SDCZoneFunctor(const SmoothDoubleCubic<Real>& curve,
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
        const SmoothDoubleCubic<Real>& curve_;
        Real xZone_;
        Real fShift_;
        unsigned power_;
    };
}

#endif // ASE_SMOOTHDOUBLECUBIC_HH_
