#ifndef ASE_OPATGAUSSIAN_HH_
#define ASE_OPATGAUSSIAN_HH_

#include <limits>
#include <cassert>
#include <algorithm>
#include <stdexcept>

#include "ase/Gaussian.hh"
#include "ase/specialFunctions.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/findRootUsingBisections.hh"

namespace ase {
    template<class T>
    class OPATGaussian : public AbsLocationScaleFamily
    {
    public:
        typedef T Transform;

        static const bool isFullOPAT = true;

        inline OPATGaussian(const double i_median,
                            const double i_sigmaPlus,
                            const double i_sigmaMinus,
                            const Transform& tr)
            : AbsLocationScaleFamily(i_median, (std::abs(i_sigmaPlus) +
                                                std::abs(i_sigmaMinus))/2.0),
              sigmaPlus_(i_sigmaPlus),
              sigmaMinus_(i_sigmaMinus),
              tr_(tr),
              g_(0.0, 1.0),
              cumulantsCalculated_(false),
              convex_(false)
        {
            initialize();
        }

        inline virtual ~OPATGaussian() override {}

        inline double sigmaPlus() const {return sigmaPlus_;}
        inline double sigmaMinus() const {return sigmaMinus_;}

        inline virtual void setScale(const double s) override
        {
            if (s <= 0.0) throw std::invalid_argument(
                "In ase::OPATGaussian::setScale: "
                "scale parameter must be positive");
            if (sigmaPlus_ != sigmaMinus_)
            {
                const double oldScale = scale();
                const double factor = s/oldScale;
                sigmaPlus_ *= factor;
                sigmaMinus_ *= factor;
                AbsLocationScaleFamily::setScale((std::abs(sigmaPlus_) +
                                                  std::abs(sigmaMinus_))/2.0);
            }
            else
                AbsLocationScaleFamily::setScale(s);
        }

        inline virtual bool isUnimodal() const override
            {return !tr_.hasExtremum();}

    protected:
        void initialize()
        {
            if (sigmaPlus_ == 0.0 && sigmaMinus_ == 0.0)
                throw std::invalid_argument(
                    "In ase::OPATGaussian::initialize: "
                    "both sigmas can not simultaneously be 0");
            if (tr_.isFlat()) throw std::invalid_argument(
                "In ase::OPATGaussian::initialize: "
                "invalid combination of arguments, "
                "leads to a Dirac delta component");
            if (tr_.hasExtremum())
                convex_ = tr_.secondDerivative(tr_.extremum().first) > 0.0;
            determineLimits();
        }

        double sigmaPlus_;
        double sigmaMinus_;
        double xmin_;
        double xmax_;

        Transform tr_;
        Gaussian g_;

        mutable double cumulants_[5];
        mutable bool cumulantsCalculated_;
        bool convex_;

    private:
        inline void determineLimits()
        {
            xmin_ = inverseGaussCdf(0.0);
            xmax_ = inverseGaussCdf(1.0);
            if (sigmaPlus_ != sigmaMinus_)
            {
                const double tmp1 = tr_(xmin_);
                const double tmp2 = tr_(xmax_);
                double tmp3 = 0.0;
                if (tr_.hasExtremum())
                    tmp3 = tr_.extremum().second;
                xmin_ = std::min(std::min(tmp1, tmp2), tmp3);
                xmax_ = std::max(std::max(tmp1, tmp2), tmp3);
            }
        }

        virtual long double calculateMoment(
            long double mu, unsigned power) const = 0;

        inline void calculateCumulants() const
        {
            cumulants_[0] = 1.0;
            const long double mu = calculateMoment(0.0L, 1U);
            cumulants_[1] = mu;
            const long double var = calculateMoment(mu, 2U);
            cumulants_[2] = var;
            cumulants_[3] = calculateMoment(mu, 3U);
            cumulants_[4] = calculateMoment(mu, 4U) - 3.0L*var*var;
            cumulantsCalculated_ = true;
        }

        inline virtual double unscaledDensity(const double x) const override
        {
            if (sigmaPlus_ != sigmaMinus_)
            {
                if (x <= xmin_ || x >= xmax_)
                    return 0.0;
                long double u[2];
                const unsigned nRoots = tr_.inverse(x, u);
                long double sum = 0.0L;
                for (unsigned i=0; i<nRoots; ++i)
                    sum += g_.density(u[i])/std::abs(tr_.derivative(u[i]));
                return sum;
            }
            else
                return g_.density(x);
        }

        inline virtual double unscaledDensityDerivative(const double x) const override
        {
            if (sigmaPlus_ != sigmaMinus_)
            {
                if (x <= xmin_ || x >= xmax_)
                    return 0.0;
                long double uvals[2];
                const unsigned nRoots = tr_.inverse(x, uvals);
                long double sum = 0.0L;
                for (unsigned i=0; i<nRoots; ++i)
                {
                    const double u = uvals[i];
                    const double der = tr_.derivative(u);
                    const double absder = std::abs(der);
                    const double sder = tr_.secondDerivative(u);
                    const double dens = g_.density(u);

                    sum += (g_.densityDerivative(u) - dens*sder/der)/der/absder;
                }
                return sum;
            }
            else
                return g_.densityDerivative(x);
        }

        inline virtual double unscaledCdf(const double x) const override
        {
            if (sigmaPlus_ != sigmaMinus_)
            {
                if (x <= xmin_)
                    return 0.0;
                if (x >= xmax_)
                    return 1.0;
                long double u[2];
                const unsigned nRoots = tr_.inverse(x, u);
                assert(nRoots == 1U || nRoots == 2U);
                if (nRoots == 1U)
                    return g_.cdf(u[0]);
                else
                {
                    assert(u[1] >= u[0]);
                    if (convex_)
                        return g_.cdf(u[1]) - g_.cdf(u[0]);
                    else
                        return g_.cdf(u[0]) + g_.exceedance(u[1]);
                }
            }
            else
                return g_.cdf(x);
        }

        inline virtual double unscaledExceedance(const double x) const override
        {
            if (sigmaPlus_ != sigmaMinus_)
            {
                if (x <= xmin_)
                    return 1.0;
                if (x >= xmax_)
                    return 0.0;
                long double u[2];
                const unsigned nRoots = tr_.inverse(x, u);
                assert(nRoots == 1U || nRoots == 2U);
                if (nRoots == 1U)
                    return g_.exceedance(u[0]);
                else
                {
                    assert(u[1] >= u[0]);
                    if (convex_)
                        return g_.cdf(u[0]) + g_.exceedance(u[1]);
                    else
                        return g_.cdf(u[1]) - g_.cdf(u[0]);
                }
            }
            else
                return g_.exceedance(x);
        }

        inline virtual double unscaledQuantile(const double r1) const override
        {
            if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
                "In ase::OPATGaussian::unscaledQuantile: "
                "cdf argument outside of [0, 1] interval");
            if (r1 == 0.0)
                return xmin_;
            if (r1 == 1.0)
                return xmax_;
            if (sigmaPlus_ != sigmaMinus_)
            {
                const double tol = 2.0*std::numeric_limits<double>::epsilon();
                double q = 0.0;
                const bool status = findRootUsingBisections(
                    UnscaledCdfFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
                if (!status) throw std::runtime_error(
                    "In ase::OPATGaussian::unscaledQuantile: "
                    "root finding failed");
                return q;
            }
            else
                return g_.quantile(r1);
        }

        inline virtual double unscaledInvExceedance(const double r1) const override
        {
            if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
                "In ase::OPATGaussian::unscaledInvExceedance: "
                "exceedance argument outside of [0, 1] interval");
            if (r1 == 1.0)
                return xmin_;
            if (r1 == 0.0)
                return xmax_;
            if (sigmaPlus_ != sigmaMinus_)
            {
                const double tol = 2.0*std::numeric_limits<double>::epsilon();
                double q = 0.0;
                const bool status = findRootUsingBisections(
                    UnscaledExceedanceFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
                if (!status) throw std::runtime_error(
                    "In ase::OPATGaussian::unscaledInvExceedance: "
                    "root finding failed");
                return q;
            }
            else
                return g_.invExceedance(r1);
        }

        inline virtual double unscaledCumulant(const unsigned n) const override
        {
            if (n > 4U) throw std::invalid_argument(
                "In ase::OPATGaussian::unscaledCumulant: "
                "only four leading cumulants are implemented");
            if (sigmaPlus_ != sigmaMinus_)
            {
                if (!cumulantsCalculated_)
                    calculateCumulants();
                return cumulants_[n];
            }
            else
                return g_.cumulant(n);
        }

        inline virtual double unscaledRandom(AbsRNG& gen) const override
        {
            const double x = g_.random(gen);
            if (sigmaPlus_ != sigmaMinus_)
                return tr_(x);
            else
                return x;
        }
    };
}

#endif // ASE_OPATGAUSSIAN_HH_
