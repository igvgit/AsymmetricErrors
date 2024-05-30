#ifndef ASE_DISTRIBUTIONFUNCTORS1D_HH_
#define ASE_DISTRIBUTIONFUNCTORS1D_HH_

#include <cmath>
#include <stdexcept>

#include "ase/AbsLocationScaleFamily.hh"

namespace ase {
    class DensityFunctor1D
    {
    public:
        inline explicit DensityFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.density(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class DensityDerivativeFunctor1D
    {
    public:
        inline explicit DensityDerivativeFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.densityDerivative(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class ShiftedDensityFunctor1D
    {
    public:
        inline explicit ShiftedDensityFunctor1D(
            const AbsDistributionModel1D& fcn, const double shift,
            const bool flipSign = false)
            : fcn_(fcn), shift_(shift), factor_(flipSign ? -1.0 : 1.0) {}

        inline double operator()(const double x) const
            {return fcn_.density(factor_*(x - shift_));}

    private:
        const AbsDistributionModel1D& fcn_;
        double shift_;
        double factor_;
    };

    class CdfFunctor1D
    {
    public:
        inline explicit CdfFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.cdf(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class ExceedanceFunctor1D
    {
    public:
        inline explicit ExceedanceFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.exceedance(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class InvExceedanceFunctor1D
    {
    public:
        inline explicit InvExceedanceFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.invExceedance(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class QuantileFunctor1D
    {
    public:
        inline explicit QuantileFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.quantile(x);}

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class EntropyFunctor1D
    {
    public:
        inline explicit EntropyFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
        {
            const double d = fcn_.density(x);
            if (d <= 0.0)
                return 0.0;
            else
                return -d*log(d);
        }

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class LogDensityFunctor1D
    {
    public:
        inline explicit LogDensityFunctor1D(const AbsDistributionModel1D& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
        {
            const double d = fcn_.density(x);
            if (d <= 0.0) throw std::invalid_argument(
                "In ase::LogDensityFunctor1D::operator(): "
                "argument outside of density support");
            return log(d);
        }

    private:
        const AbsDistributionModel1D& fcn_;
    };

    class MomentFunctor1D
    {
    public:
        inline MomentFunctor1D(const AbsDistributionModel1D& fcn,
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
            return deltaPow*fcn_.density(x);
        }

    private:
        const AbsDistributionModel1D& fcn_;
        double mu0_;
        unsigned n_;
    };

    // The following functor is useful with Gauss-Hermite integration
    class RatioMomentFunctor1D
    {
    public:
        inline RatioMomentFunctor1D(const AbsDistributionModel1D& fcnNum,
                                    const AbsDistributionModel1D& fcnDenom,
                                    const double mu0, const unsigned n)
            : fcnNum_(fcnNum), fcnDenom_(fcnDenom), mu0_(mu0), n_(n) {}

        inline double operator()(const double x) const
        {
            const double num = fcnNum_.density(x);
            if (num == 0.0)
                return 0.0;

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
            if (denom <= 0.0) throw std::invalid_argument(
                "In ase::RatioMomentFunctor1D::operator(): "
                "argument outside of denominator density support");

            return num/denom*deltaPow;
        }

    private:
        const AbsDistributionModel1D& fcnNum_;
        const AbsDistributionModel1D& fcnDenom_;
        double mu0_;
        unsigned n_;
    };

    class UnscaledCdfFunctor1D
    {
    public:
        inline explicit UnscaledCdfFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledCdf(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledExceedanceFunctor1D
    {
    public:
        inline explicit UnscaledExceedanceFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledExceedance(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledQuantileFunctor1D
    {
    public:
        inline explicit UnscaledQuantileFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledQuantile(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledInvExceedanceFunctor1D
    {
    public:
        inline explicit UnscaledInvExceedanceFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledInvExceedance(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledDensityFunctor1D
    {
    public:
        inline explicit UnscaledDensityFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledDensity(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledDensityDerivativeFunctor1D
    {
    public:
        inline explicit UnscaledDensityDerivativeFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
            {return fcn_.unscaledDensityDerivative(x);}

    private:
        const AbsLocationScaleFamily& fcn_;
    };

    class UnscaledMomentFunctor1D
    {
    public:
        inline UnscaledMomentFunctor1D(const AbsLocationScaleFamily& fcn,
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
            return deltaPow*fcn_.unscaledDensity(x);
        }

    private:
        const AbsLocationScaleFamily& fcn_;
        double mu0_;
        unsigned n_;
    };

    class UnscaledEntropyFunctor1D
    {
    public:
        inline explicit UnscaledEntropyFunctor1D(const AbsLocationScaleFamily& fcn)
            : fcn_(fcn) {}

        inline double operator()(const double x) const
        {
            const double d = fcn_.unscaledDensity(x);
            if (d <= 0.0)
                return 0.0;
            else
                return -d*log(d);
        }

    private:
        const AbsLocationScaleFamily& fcn_;
    };
}

#endif // ASE_DISTRIBUTIONFUNCTORS1D_HH_
