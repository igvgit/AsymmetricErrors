#ifndef ASE_PARABOLICRAILWAYCURVE_HH_
#define ASE_PARABOLICRAILWAYCURVE_HH_

#include <cmath>
#include <limits>
#include <cassert>
#include <stdexcept>

#include "ase/Interval.hh"
#include "ase/TransitionCubic.hh"
#include "ase/findRootUsingBisections.hh"

namespace ase {
    /**
    // Something like a railway curve, smoothly transitioning
    // from one straight line to another, with continuous second
    // derivative. In typical railway curves used to turn trains,
    // the central section is circular. Here, it is parabolic.
    */
    template<typename Real>
    class ParabolicRailwayCurve
    {
    public:
        typedef Real value_type;

        inline ParabolicRailwayCurve(const Real i_sigmaPlus,
                                     const Real i_sigmaMinus,
                                     const Real i_hleft,
                                     const Real i_hright,
                                     const Real scaleNorm = 1.0)
            : sigmaPlus_(i_sigmaPlus/scaleNorm),
              sigmaMinus_(i_sigmaMinus/scaleNorm),
              abshleft_(std::abs(i_hleft)),
              abshright_(std::abs(i_hright)),
              sigma_((sigmaPlus_ + sigmaMinus_)/2.0),
              alpha_((sigmaPlus_ - sigmaMinus_)/2.0),
              left_(-1.0, -abshleft_, -sigmaMinus_, sigma_-2.0*alpha_, 2.0*alpha_),
              right_(1.0, abshright_, sigmaPlus_, sigma_+2.0*alpha_, 2.0*alpha_),
              leftDer_(left_.derivative(-1.0 - abshleft_)),
              leftValue_(left_(-1.0 - abshleft_)),
              rightDer_(right_.derivative(1.0 + abshright_)),
              rightValue_(right_(1.0 + abshright_))
        {
            if (scaleNorm <= 0.0) throw std::invalid_argument(
                "In ase::ParabolicRailwayCurve constructor: "
                "scale normalization factor must be positive");
            extremumArg_ = determineExtremumLocation();
            extremumValue_ = (*this)(extremumArg_);
        }

        inline Real hleft() const {return abshleft_;}
        inline Real hright() const {return abshright_;}
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
            if (x < -1.0 - abshleft_)
                return leftValue_ + leftDer_*(x + 1.0 + abshleft_);
            else if (x < -1.0)
                return left_(x);
            else if (x < 1.0)
                return x*(sigma_ + alpha_*x);
            else if (x < 1.0 + abshright_)
                return right_(x);
            else
                return rightValue_ + rightDer_*(x - 1.0 - abshright_);
        }

        inline Real derivative(const Real x) const
        {
            if (x < -1.0 - abshleft_)
                return leftDer_;
            else if (x < -1.0)
                return left_.derivative(x);
            else if (x < 1.0)
                return sigma_+2.0*alpha_*x;
            else if (x < 1.0 + abshright_)
                return right_.derivative(x);
            else
                return rightDer_;
        }

        inline Real secondDerivative(const Real x) const
        {
            if (x < -1.0 - abshleft_)
                return 0.0;
            else if (x < -1.0)
                return left_.secondDerivative(x);
            else if (x < 1.0)
                return 2.0*alpha_;
            else if (x < 1.0 + abshright_)
                return right_.secondDerivative(x);
            else
                return 0.0;
        }

        inline Real zoneContinuation(const Real xZone, const Real x) const
        {
            if (xZone < -1.0 - abshleft_)
                return leftValue_ + leftDer_*(x + 1.0 + abshleft_);
            else if (xZone < -1.0)
                return left_(x);
            else if (xZone < 1.0)
                return x*(sigma_ + alpha_*x);
            else if (xZone < 1.0 + abshright_)
                return right_(x);
            else
                return rightValue_ + rightDer_*(x - 1.0 - abshright_);
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

            const bool increasing = derivative(0) > 0.0;
            if ((increasing && y >= rightValue_) || (!increasing && y <= rightValue_))
            {
                solutions[0] = (y - rightValue_)/rightDer_ + abshright_ + 1.0;
                return 1U;
            }
            if ((increasing && y <= leftValue_) || (!increasing && y >= leftValue_))
            {
                solutions[0] = (y - leftValue_)/leftDer_ - abshleft_ - 1.0;
                return 1U;
            }

            const Interval<Real> yrange(leftValue_, rightValue_, OPEN_INTERVAL);
            assert(yrange.contains(y));
            const bool status = findRootUsingBisections(
                *this, y, -1.0 - abshleft_, 1.0 + abshright_, tol, &solutions[0]);
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

            const bool isMaximum = secondDerivative(0) < 0.0;
            if ((isMaximum && y > extremumValue_) || (!isMaximum && y < extremumValue_))
                return 0U;

            // Find the solution to the left of the extremum
            if ((isMaximum && y <= leftValue_) || (!isMaximum && y >= leftValue_))
                solutions[0] = (y - leftValue_)/leftDer_ - abshleft_ - 1.0;
            else
            {
                const Interval<Real> yrange(extremumValue_, leftValue_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, -1.0 - abshleft_, extremumArg_, tol, &solutions[0]);
                assert(status);
            }

            // Find the solution to the right of the extremum
            if ((isMaximum && y <= rightValue_) || (!isMaximum && y >= rightValue_))
                solutions[1] = (y - rightValue_)/rightDer_ + abshright_ + 1.0;
            else
            {
                const Interval<Real> yrange(extremumValue_, rightValue_, OPEN_INTERVAL);
                assert(yrange.contains(y));
                const bool status = findRootUsingBisections(
                    *this, y, extremumArg_, 1.0 + abshright_, tol, &solutions[1]);
                assert(status);
            }

            return 2U;
        }

        inline Real determineExtremumLocation() const
        {
            if (hasExtremum() && alpha_)
            {
                const Real parabolicExtr = -sigma_/2/alpha_;
                if (parabolicExtr >= -1.0 && parabolicExtr <= 1.0)
                    return parabolicExtr;
                if (left_.hasExtremum())
                {
                    assert(!right_.hasExtremum());
                    return left_.extremum().first;
                }
                if (right_.hasExtremum())
                {
                    assert(!left_.hasExtremum());
                    return right_.extremum().first;
                }
                throw std::runtime_error(
                    "In ase::ParabolicRailwayCurve::determineExtremumLocation: "
                    "extremum search failed. This is a bug. "
                    "Please report in a reproducible manner.");
            }
            return 0.0;
        }

        Real sigmaPlus_;
        Real sigmaMinus_;
        Real abshleft_;
        Real abshright_;
        Real sigma_;
        Real alpha_;
        TransitionCubic<Real> left_;
        TransitionCubic<Real> right_;
        Real leftDer_;
        Real leftValue_;
        Real rightDer_;
        Real rightValue_;
        Real extremumArg_;
        Real extremumValue_;
    };

    template<typename Real>
    class RailwayZoneFunctor
    {
    public:
        inline RailwayZoneFunctor(const ParabolicRailwayCurve<Real>& curve,
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
        const ParabolicRailwayCurve<Real>& curve_;
        Real xZone_;
        Real fShift_;
        unsigned power_;
    };
}

#endif // ASE_PARABOLICRAILWAYCURVE_HH_
