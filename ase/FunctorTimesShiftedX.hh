#ifndef ASE_FUNCTORTIMESSHIFTEDX_HH_
#define ASE_FUNCTORTIMESSHIFTEDX_HH_

#include <cmath>
#include <stdexcept>

#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    template<class Functor>
    class FunctorTimesShiftedXHelper
    {
    public:
        inline FunctorTimesShiftedXHelper(const Functor& fcn,
                                          const double mu0, const unsigned n)
            : fcn_(fcn), mu0_(mu0), n_(n) {}

        inline double operator()(const double x) const
        {
            double deltaPow = 1.0;
            if (n_)
            {
                const double delta = x - mu0_;
                const double delta2 = delta*delta;
                switch (n_)
                {
                case 1U:
                    deltaPow = delta;
                    break;
                case 2U:
                    deltaPow = delta2;
                    break;
                case 3U:
                    deltaPow = delta2*delta;
                    break;
                case 4U:
                    deltaPow = delta2*delta2;
                    break;
                default:
                    deltaPow = std::pow(delta, n_);
                }
            }
            return deltaPow*fcn_(x);
        }

    private:
        const Functor& fcn_;
        double mu0_;
        unsigned n_;
    };

    template<class Functor>
    inline FunctorTimesShiftedXHelper<Functor> FunctorTimesShiftedX(
        const Functor& fcn, const double mu0, const unsigned n)
    {
        return FunctorTimesShiftedXHelper<Functor>(fcn, mu0, n);
    }

    template<class Functor>
    class FunctorTimesShiftedXRatioHelper
    {
    public:
        inline FunctorTimesShiftedXRatioHelper(
            const Functor& fcn, const AbsDistributionModel1D& fcnDenom,
            const double mu0, const unsigned n)
            : fcn_(fcn), fcnDenom_(fcnDenom), mu0_(mu0), n_(n) {}

        inline double operator()(const double x) const
        {
            double deltaPow = 1.0;
            if (n_)
            {
                const double delta = x - mu0_;
                const double delta2 = delta*delta;
                switch (n_)
                {
                case 1U:
                    deltaPow = delta;
                    break;
                case 2U:
                    deltaPow = delta2;
                    break;
                case 3U:
                    deltaPow = delta2*delta;
                    break;
                case 4U:
                    deltaPow = delta2*delta2;
                    break;
                default:
                    deltaPow = std::pow(delta, n_);
                }
            }

            const double denom = fcnDenom_.density(x);
            if (denom == 0.0) throw std::runtime_error(
                "In ase::FunctorTimesShiftedXRatioHelper::operator(): "
                "division by zero encountered");

            return fcn_(x)/denom*deltaPow;
        }

    private:
        const Functor& fcn_;
        const AbsDistributionModel1D& fcnDenom_;
        double mu0_;
        unsigned n_;
    };

    template<class Functor>
    inline FunctorTimesShiftedXRatioHelper<Functor> FunctorTimesShiftedXRatio(
        const Functor& fcn, const AbsDistributionModel1D& fcnDenom,
        const double mu0, const unsigned n)
    {
        return FunctorTimesShiftedXRatioHelper<Functor>(fcn, fcnDenom, mu0, n);
    }
}

#endif // ASE_FUNCTORTIMESSHIFTEDX_HH_
