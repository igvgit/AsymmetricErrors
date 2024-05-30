#include <algorithm>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/DistributionModels1D.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/GaussHermiteQuadrature.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/DistributionModel1DCopy.hh"
#include "ase/InterpolatedDensity1D.hh"
#include "ase/TabulatedDensity1D.hh"
#include "ase/EquidistantGrid.hh"
#include "ase/SymmetricBetaGaussian.hh"
#include "ase/LegendreDistro1D.hh"
#include "ase/MixtureModel1D.hh"
#include "ase/TruncatedDistribution1D.hh"
#include "ase/statUtils.hh"

using namespace ase;
using namespace std;

namespace {
    typedef vector<double> Dvector;

    class GaussianDensityDerivative
    {
    public:
        inline GaussianDensityDerivative(const Gaussian& g)
            : mu_(g.location()), sigma_(g.scale()), g_(g) {}

        inline double operator()(const double x) const
        {
            return g_.density(x)*(mu_ - x)/sigma_/sigma_;
        }

    private:
        double mu_;
        double sigma_;
        Gaussian g_;
    };

    class LogJohnsonSuInverseTransform
    {
    public:
        inline LogJohnsonSuInverseTransform(const JohnsonSu& su)
            : su_(su)
        {
            params_[0] = su.getDelta();
            params_[1] = su.getLambda();
            params_[2] = su.getGamma();
            params_[3] = su.getXi();
            loc_ = su.location();
            scale_ = su.scale();
        }

        inline double operator()(const double y) const
        {
            const double x = params_[3] + params_[1]*sinh((y - params_[2])/params_[0]);
            return log(su_.density(x*scale_ + loc_));
        }

    private:
        double params_[4];
        double loc_;
        double scale_;
        const JohnsonSu& su_;
    };

    double num_density_deriv(const AbsDistributionModel1D& d, const double x,
                             const double h)
    {
        assert(h);
        const volatile double xplus = x + h;
        const volatile double xminus = x - h;
        return (d.density(xplus) - d.density(xminus))/(xplus - xminus);
    }

    void descent_test(const AbsDistributionModel1D& d, const double eps)
    {
        if (d.isUnimodal())
        {
            const bool right[2] = {true, false};
            const double mode = d.mode();
            const double modeDensity = d.density(mode);
            for (unsigned idir=0; idir<2; ++idir)
            {
                const double step = (right[idir] ? 1.0 : -1.0)*
                    d.descentDelta(right[idir]);
                const double dens = d.density(mode + step);
                const double deltaL = log(modeDensity/dens);
                CHECK_CLOSE(0.5, deltaL, eps);
            }
        }
    }

    void standard_test(const AbsDistributionModel1D& d, const double range,
                       const double eps, const bool testDerivs=true,
                       const bool testDescent=true)
    {
        const double cdf0 = d.cdf(0.0);
        for (unsigned i=0; i<100; ++i)
        {
            const double r = 2*range*(test_rng() - 0.5);
            const double cdfr = d.cdf(r);
            if (cdfr > 0.0 && cdfr < 1.0)
            {
                CHECK_CLOSE(r, d.quantile(cdfr), eps);
                const double integ = simpson_integral(
                    DensityFunctor1D(d), 0.0, r);
                CHECK_CLOSE(integ, cdfr-cdf0, eps);
                const double deriv = num_density_deriv(d, r, 1.e-5*range);
                if (testDerivs)
                    CHECK_CLOSE(deriv, d.densityDerivative(r), 10*eps);
            }
            const double excr = d.exceedance(r);
            if (excr > 0.0 && excr < 1.0)
            {
                CHECK_CLOSE(1.0, cdfr + excr, eps);
                CHECK_CLOSE(r, d.invExceedance(excr), eps);
            }
        }
        if (testDescent)
            descent_test(d, eps);
    }

    void standard_test_2(const AbsDistributionModel1D& d, const double eps)
    {
        const double cdf0 = d.cdf(0.0);
        for (unsigned i=0; i<100; ++i)
        {
            const double r = test_rng();
            const double cdfr = d.cdf(r);
            if (cdfr > 0.0 && cdfr < 1.0)
            {
                CHECK_CLOSE(r, d.quantile(cdfr), eps);
                const double integ = simpson_integral(
                    DensityFunctor1D(d), 0.0, r, 5000);
                CHECK_CLOSE(integ, cdfr-cdf0, eps);
            }
            const double excr = d.exceedance(r);
            if (excr > 0.0 && excr < 1.0)
            {
                CHECK_CLOSE(1.0, cdfr + excr, eps);
                CHECK_CLOSE(r, d.invExceedance(excr), eps);
            }
        }
    }

    template<class Distro>
    void cumulant_test(const double* cum, const unsigned ncum, const double eps)
    {
        assert(cum);
        assert(ncum > 1U);

        const Dvector cumulants(cum, cum+ncum);
        const Distro d(cumulants);
        for (unsigned i=0; i<ncum; ++i)
            CHECK_CLOSE(cum[i], d.cumulant(i+1U), eps);
    }

    void cumulant_test_sbg(const double* cum, const unsigned ncum, const double eps,
                           const unsigned p, const double h)
    {
        assert(cum);
        assert(ncum > 1U);

        const Dvector cumulants(cum, cum+ncum);
        const SymmetricBetaGaussian d(cumulants, p, h);
        for (unsigned i=0; i<ncum; ++i)
            CHECK_CLOSE(cum[i], d.cumulant(i+1U), eps);
    }

    template<class Distro>
    void entropy_test(const Distro& d, const double xmin, const double xmax,
                      const unsigned quadPoints, const unsigned nSplit,
                      const double eps)
    {
        const GaussLegendreQuadrature glq(quadPoints);
        const EntropyFunctor1D efcn(d);
        const double numEntropy = glq.integrate(efcn, xmin, xmax, nSplit);
        CHECK_CLOSE(numEntropy, d.entropy(), eps);
    }

    template<class Distro>
    void median_sigmas_test(const double median, const double sigPlus,
                            const double sigMinus, const double eps)
    {
        std::unique_ptr<Distro> d(
            Distro::fromQuantiles(median, sigPlus, sigMinus));
        CHECK_CLOSE(median, d->quantile(0.5), eps);
        CHECK_CLOSE(median-sigMinus, d->quantile(0.158655253931457051), eps);
        CHECK_CLOSE(median+sigPlus, d->quantile(0.841344746068542949), eps);
    }

    void median_sigmas_test_sbg(const double median, const double sigPlus,
                                const double sigMinus, const double eps,
                                const unsigned p, const double h)
    {
        std::unique_ptr<SymmetricBetaGaussian> d(
            SymmetricBetaGaussian::fromQuantiles(median, sigPlus, sigMinus, p, h));
        CHECK_CLOSE(median, d->quantile(0.5), eps);
        CHECK_CLOSE(median-sigMinus, d->quantile(0.158655253931457051), eps);
        CHECK_CLOSE(median+sigPlus, d->quantile(0.841344746068542949), eps);
    }

    TEST(Gaussian_1)
    {
        const Gaussian g(0., 1.);
        double value = simpson_integral(DensityFunctor1D(g), -6, 6);
        CHECK_CLOSE(1.0, value, 1.e-8);
        standard_test(g, 6, 1.e-7);
        standard_test_2(g, 1.e-7);
        entropy_test(g, -10.0, 10.0, 4, 2000, 1.0e-12);

        const Gaussian g7(3., 7.);
        entropy_test(g7, -67.0, 73.0, 4, 2000, 1.0e-12);

        CHECK_CLOSE(7.61985302416e-24, g.cdf(-10.0), 1.e-34);
        CHECK_CLOSE(7.61985302416e-24, g.exceedance(10.0), 1.e-34);
        CHECK_CLOSE(2.75362411861e-89, g.cdf(-20.0), 1.e-99);
        CHECK_CLOSE(2.75362411861e-89, g.exceedance(20.0), 1.e-99);

        double cum1[] = {0.1, 1.2};
        cumulant_test<Gaussian>(cum1, sizeof(cum1)/sizeof(cum1[0]), 1.0e-15);

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig = test_rng() + 0.1;
            median_sigmas_test<Gaussian>(med, sig, sig, 1.0e-15);
        }
    }

    TEST(TruncatedDistribution1D_1)
    {
        const Gaussian g(0., 1.);
        const TruncatedDistribution1D tr(g, -2.0, 2.0);
        standard_test(tr, 2.0, 1.e-7, true, false);
        standard_test_2(tr, 1.e-7);
    }

    TEST(FechnerDistribution_1)
    {
        const double h = 1.0e-5;
        const double eps = 1.0e-12;
        const double A = sqrt(2.0/M_PI);

        for (unsigned i=0; i<100; ++i)
        {
            const double mu = test_rng() - 0.5;
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            const FechnerDistribution fd(mu, sig1, sig2);

            const double x = test_rng() - 0.5;
            const double del = x > mu ? (x - mu)/sig1 : (x - mu)/sig2;
            const double dens = A/(sig1 + sig2)*exp(-del*del/2.0);
            CHECK_CLOSE(dens, fd.density(x), eps);

            const double xplus = x + h;
            const double xminus = x - h;
            const double der = (fd.density(xplus) - fd.density(xminus))/(xplus - xminus);
            CHECK_CLOSE(der, fd.densityDerivative(x), eps/h);

            const double sigmin = std::min(sig1, sig2);
            standard_test(fd, 6*sigmin, 1.0e-5);
            standard_test_2(fd, eps);

            const double twoOverPi = 2.0/M_PI;
            const double sqrTwoOverPi = sqrt(twoOverPi);
            const double expMean = mu + sqrTwoOverPi*(sig1 - sig2);
            CHECK_CLOSE(expMean, fd.cumulant(1), eps);
            const double expVar = (1.0 - twoOverPi)*(sig1 - sig2)*(sig1 - sig2) + sig1*sig2;
            CHECK_CLOSE(expVar, fd.cumulant(2), eps);
            const double expSkew = sqrTwoOverPi*(sig1 - sig2)*((2.0*twoOverPi - 1.0)*(sig1 - sig2)*(sig1 - sig2) + sig1*sig2);
            CHECK_CLOSE(expSkew, fd.cumulant(3), eps);
        }
    }

    TEST(FechnerDistribution_2)
    {
        const double eps = 1.0e-12;
        const double maxSkew = M_SQRT2*(4.0 - M_PI)/pow(M_PI - 2.0, 1.5);

        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<1000; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<FechnerDistribution>(cum4, sizeof(cum4)/sizeof(cum4[0]), eps);
            cumulant_test<FechnerDistribution>(cum4, 2U, eps);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 3.0;
            const double sig2 = test_rng() + 3.0;
            median_sigmas_test<FechnerDistribution>(med, sig1, sig2, eps);
        }
    }

    TEST(DimidiatedGaussian_1)
    {
        const double eps = 1.0e-15;

        const DimidiatedGaussian dg(0.0, 1.2, 1.3);
        standard_test(dg, 6, 1.e-4);
        standard_test_2(dg, 1.e-4);

        const Gaussian g(0.0, 1.0);
        const double cdfm1 = g.cdf(-1.0);
        const double cdf1 = g.cdf(1.0);
        CHECK_CLOSE(cdfm1, dg.cdf(-dg.sigmaMinus()), eps);
        CHECK_CLOSE(cdf1, dg.cdf(dg.sigmaPlus()), eps);

        // The following set of cumulants should result in sigmaMinus = 0.
        // DimidiatedGaussian constructor should throw an exception.
        double cum1[] = {0.5, 1.0, (M_PI + 2.0)/pow(M_PI - 1.0, 1.5)};
        bool caught_exception = false;
        try {
            cumulant_test<DimidiatedGaussian>(cum1, sizeof(cum1)/sizeof(cum1[0]), eps);
        }
        catch (const std::invalid_argument& e) {
            caught_exception = true;
        }
        CHECK(caught_exception);

        const double maxSkew = 2.0/sqrt(2.0*M_PI - 5.0) - 0.01;
        double cum4[] = {0.1, 1.0, 0.0};
        for (unsigned i=0; i<1000; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<DimidiatedGaussian>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;
            median_sigmas_test<DimidiatedGaussian>(med, sig1, sig2, 1.0e-15);
        }
    }

    TEST(DistortedGaussian_1)
    {
        const double eps = 1.0e-15;

        const DistortedGaussian dg(0.0, 1.2, 1.3);
        standard_test(dg, 4, 1.e-8);
        standard_test_2(dg, 1.e-8);

        const Gaussian g(0.0, 1.0);
        const double cdfm1 = g.cdf(-1.0);
        const double cdf1 = g.cdf(1.0);
        CHECK_CLOSE(cdfm1, dg.cdf(-dg.sigmaMinus()), eps);
        CHECK_CLOSE(cdf1, dg.cdf(dg.sigmaPlus()), eps);

        double cum1[] = {0.5, 1.2, 2.5};
        cumulant_test<DistortedGaussian>(cum1, sizeof(cum1)/sizeof(cum1[0]), eps);

        double cum2[] = {0.5, 1.2, -2.3};
        cumulant_test<DistortedGaussian>(cum2, sizeof(cum2)/sizeof(cum2[0]), eps);

        double cum3[] = {0.5, 1.2, 0.0};
        cumulant_test<DistortedGaussian>(cum3, sizeof(cum3)/sizeof(cum3[0]), eps);

        const double maxSkew = 2.0*M_SQRT2;
        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<1000; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<DistortedGaussian>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        double cum5[] = {0.5, 1.0, 14.0/(3.0*sqrt(3.0))};
        cumulant_test<DistortedGaussian>(cum5, sizeof(cum5)/sizeof(cum5[0]), 1.0e-14);

        double cum6[] = {0.5, 1.0, -14.0/(3.0*sqrt(3.0))};
        cumulant_test<DistortedGaussian>(cum6, sizeof(cum6)/sizeof(cum6[0]), 1.0e-14);

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            median_sigmas_test<DistortedGaussian>(med, sig1, sig2, 1.0e-12);
        }
    }

    TEST(RailwayGaussian_1)
    {
        const double eps = 1.0e-15;
        const double maxSkew = 2.429336334952816;

        const RailwayGaussian dg(0.0, 1.2, 1.3, 1.0, 1.2);
        standard_test(dg, 4, 1.e-8);
        standard_test_2(dg, 1.e-8);

        const Gaussian g(0.0, 1.0);
        const double cdfm1 = g.cdf(-1.0);
        const double cdf1 = g.cdf(1.0);
        CHECK_CLOSE(cdfm1, dg.cdf(-dg.sigmaMinus()), eps);
        CHECK_CLOSE(cdf1, dg.cdf(dg.sigmaPlus()), eps);

        double cum1[] = {0.5, 1.2, 2.4};
        cumulant_test<RailwayGaussian>(cum1, sizeof(cum1)/sizeof(cum1[0]), eps);

        double cum2[] = {0.5, 1.2, -2.3};
        cumulant_test<RailwayGaussian>(cum2, sizeof(cum2)/sizeof(cum2[0]), eps);

        double cum3[] = {0.5, 1.2, 0.0};
        cumulant_test<RailwayGaussian>(cum3, sizeof(cum3)/sizeof(cum3[0]), eps);

        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<100; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<RailwayGaussian>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 0.55;
            const double sig2 = test_rng() + 0.55;
            median_sigmas_test<RailwayGaussian>(med, sig1, sig2, 1.0e-12);
        }
    }

    TEST(RailwayGaussian_2)
    {
        const double eps = 1.0e-10;
        const double maxrat = 4.0591;

        for (unsigned i=0; i<200; ++i)
        {
            const double med = test_rng() - 0.5;
            double sigp = maxrat, sigm = 1.0, sigrat = 0.0;

            if (i)
            {
                while (sigrat > maxrat || sigrat < 1.0/maxrat)
                {
                    sigp = test_rng() + 0.2;
                    sigm = test_rng() + 0.2;
                    sigrat = sigp/sigm;
                }
            }
            median_sigmas_test<RailwayGaussian>(med, sigp, sigm, eps);
        }
    }

    TEST(DoubleCubicGaussian_1)
    {
        const double eps = 1.0e-15;
        const double maxSkew = 1.88785158;

        const DoubleCubicGaussian dg(0.0, 1.2, 1.3);
        standard_test(dg, 4, 1.e-8);
        standard_test_2(dg, 1.e-8);

        const Gaussian g(0.0, 1.0);
        const double cdfm1 = g.cdf(-1.0);
        const double cdf1 = g.cdf(1.0);
        CHECK_CLOSE(cdfm1, dg.cdf(-dg.sigmaMinus()), eps);
        CHECK_CLOSE(cdf1, dg.cdf(dg.sigmaPlus()), eps);

        double cum1[] = {0.5, 1.2, 2.4};
        cumulant_test<DoubleCubicGaussian>(cum1, sizeof(cum1)/sizeof(cum1[0]), eps);

        double cum2[] = {0.5, 1.2, -2.3};
        cumulant_test<DoubleCubicGaussian>(cum2, sizeof(cum2)/sizeof(cum2[0]), eps);

        double cum3[] = {0.5, 1.2, 0.0};
        cumulant_test<DoubleCubicGaussian>(cum3, sizeof(cum3)/sizeof(cum3[0]), eps);

        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<100; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<DoubleCubicGaussian>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 0.55;
            const double sig2 = test_rng() + 0.55;
            median_sigmas_test<DoubleCubicGaussian>(med, sig1, sig2, 1.0e-12);
        }
    }

    TEST(DoubleCubicGaussian_2)
    {
        const double eps = 1.0e-10;
        const double maxrat = 6.854844;

        for (unsigned i=0; i<200; ++i)
        {
            const double med = test_rng() - 0.5;
            double sigp = maxrat, sigm = 1.0, sigrat = 0.0;

            if (i)
            {
                while (sigrat > maxrat || sigrat < 1.0/maxrat)
                {
                    sigp = test_rng() + 0.1;
                    sigm = test_rng() + 0.1;
                    sigrat = sigp/sigm;
                }
            }
            median_sigmas_test<DoubleCubicGaussian>(med, sigp, sigm, eps);
        }
    }

    TEST(SymmetricBetaGaussian_0)
    {
        for (unsigned p=1; p<5; ++p)
        {
            const SymmetricBetaGaussian sbg(0.0, 1.1, 1.3, p, 2.0);
            standard_test(sbg, 4, 1.e-8);
            standard_test_2(sbg, 1.e-8);
        }
    }

    TEST(SymmetricBetaGaussian_1)
    {
        const double eps = 1.0e-10;
        const double maxSkew = 1.0;

        const SymmetricBetaGaussian dg(0.0, 1.2, 1.3, 2U, 1.5);
        const Gaussian g(0.0, 1.0);
        const double cdfm1 = g.cdf(-1.0);
        const double cdf1 = g.cdf(1.0);
        CHECK_CLOSE(cdfm1, dg.cdf(-dg.sigmaMinus()), eps);
        CHECK_CLOSE(cdf1, dg.cdf(dg.sigmaPlus()), eps);

        double cum1[] = {0.5, 1.2, 2.4};
        cumulant_test_sbg(cum1, sizeof(cum1)/sizeof(cum1[0]), eps, 1U, 0.5 + test_rng());

        double cum2[] = {0.5, 1.2, -2.3};
        cumulant_test_sbg(cum2, sizeof(cum2)/sizeof(cum2[0]), eps, 2U, 0.5 + test_rng());

        double cum3[] = {0.5, 1.2, 0.0};
        cumulant_test_sbg(cum3, sizeof(cum3)/sizeof(cum3[0]), eps, 3U, 0.5 + test_rng());

        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<100; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test_sbg(cum4, sizeof(cum4)/sizeof(cum4[0]), eps, 2U, 0.5 + test_rng());
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            const unsigned p = 1 + test_rng()*5;
            const double h = 0.5 + test_rng();
            median_sigmas_test_sbg(med, sig1, sig2, 1.0e-12, p, h);
        }
    }

    TEST(SymmetricBetaGaussian_2)
    {
        const double eps = 1.0e-6;

        for (unsigned i=0; i<100; ++i)
        {
            const double mode = test_rng();
            const double deltaPlus = test_rng() + 2.0;
            const double deltaMinus = test_rng() + 2.0;
            const unsigned p = 1 + test_rng()*5;
            const double h = 0.5 + test_rng();
            const double deltaLnL = 0.5 + test_rng();
            // const double deltaLnL = 0.5;

            std::unique_ptr<SymmetricBetaGaussian> sbg =
                SymmetricBetaGaussian::fromModeAndDeltas(
                    mode, deltaPlus, deltaMinus, p, h, deltaLnL);

            const double m = sbg->mode();
            CHECK_CLOSE(mode, m, eps);

            const double ddr = sbg->descentDelta(true, deltaLnL);
            CHECK_CLOSE(log(sbg->density(m)/sbg->density(m + ddr)), deltaLnL, eps);

            const double ddl = sbg->descentDelta(false, deltaLnL);
            CHECK_CLOSE(log(sbg->density(m)/sbg->density(m - ddl)), deltaLnL, eps);
        }
    }

    TEST(SkewNormal_1)
    {
        const SkewNormal sn(0.0, 1.2, 0.5);
        standard_test(sn, 4, 1.e-8);
        standard_test_2(sn, 1.e-8);

        const SkewNormal sn2(0.0, 1.2, -0.5);
        // standard_test(sn2, 4, 1.e-8);
        standard_test_2(sn2, 1.e-8);

        const double maxSkew = M_SQRT2*(4.0 - M_PI)/pow(M_PI - 2.0, 1.5);
        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<1000; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<SkewNormal>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 2.0;
            const double sig2 = test_rng() + 2.0;
            median_sigmas_test<SkewNormal>(med, sig1, sig2, 1.0e-12);
        }
    }

    TEST(QVWGaussian_1)
    {
        const double mom11 = 1/(2*sqrt(M_PI));
        const double mom22 = (sqrt(3.0) + 2*M_PI)/(6*M_PI);

        for (unsigned i=0; i<100; ++i)
        {
            const QVWGaussian qvwg(0.5, 2.0, test_rng() - 0.5);

            const double mu = qvwg.locationParameter();
            const double sigma = qvwg.scaleParameter();
            const double a = qvwg.asymmetryParameter();

            const double mean = mu + sigma*a*mom11;
            CHECK_CLOSE(0.0, mean, 1.0e-15);

            const double var = mu*mu + 2*a*mu*sigma*mom11 + (sigma*sigma*(4 + a*a*(-1 + 4*mom22)))/4;
            CHECK_CLOSE(1.0, var, 1.0e-13);
        }
    }

    TEST(QVWGaussian_2)
    {
        const QVWGaussian qvwg(0.0, 1.2, 0.5);
        standard_test(qvwg, 4, 1.e-10);
        standard_test_2(qvwg, 1.e-10);

        const double maxSkew = 1.7722856727810639;
        double cum4[] = {0.0, 1.0, 0.0};
        for (unsigned i=0; i<1000; ++i)
        {
            cum4[0] = test_rng();
            cum4[1] = test_rng() + 0.1;
            cum4[2] = (test_rng() - 0.5)*2.0*maxSkew*pow(cum4[1], 1.5);
            cumulant_test<QVWGaussian>(cum4, sizeof(cum4)/sizeof(cum4[0]), 1.0e-12);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double med = test_rng() - 0.5;
            const double sig1 = test_rng() + 2.0;
            const double sig2 = test_rng() + 2.0;
            median_sigmas_test<SkewNormal>(med, sig1, sig2, 1.0e-12);
        }
    }

    TEST(EdgeworthExpansion3_1)
    {
        double cum[3] = {0.0, 1.0, 0.1};
        const EdgeworthExpansion3 expan(Dvector(cum, cum+sizeof(cum)/sizeof(cum[0])));
        standard_test(expan, 4, 1.e-7);
        standard_test_2(expan, 1.e-7);
        cumulant_test<EdgeworthExpansion3>(cum, sizeof(cum)/sizeof(cum[0]), 10.e-10);

        double cum2[3] = {1.0, 2.0, 0.15};
        cumulant_test<EdgeworthExpansion3>(cum2, sizeof(cum2)/sizeof(cum2[0]), 10.e-10);

        // Calculate the cumulants numerically
        const GaussHermiteQuadrature ghq(16);
        const EdgeworthExpansion3 ee(Dvector(cum2, cum2+sizeof(cum2)/sizeof(cum2[0])));
        const Gaussian g(cum2[0], sqrt(cum2[1]));
        const RatioMomentFunctor1D rf0(ee, g, 0.0, 1U);
        const double mean = ghq.integrateProb(cum2[0], sqrt(cum2[1]), rf0);
        double cumEval[4];
        cumEval[0] = mean;
        for (unsigned n=2; n<=4; ++n)
        {
            const RatioMomentFunctor1D rf(ee, g, cum2[0], n);
            cumEval[n-1U] = ghq.integrateProb(cum2[0], sqrt(cum2[1]), rf);
        }
        cumEval[3] -= 3.0*cumEval[1]*cumEval[1];
        for (unsigned n=0; n<4; ++n)
            CHECK_CLOSE(cum2[n], cumEval[n], 1.0e-14);
    }

    TEST(Gamma_cumulants)
    {
        const double eps = 1.0e-10;

        double cum[3] = {3.9, 2.88, 6.912};
        cumulant_test<GammaDistribution>(cum, sizeof(cum)/sizeof(cum[0]), eps);
    }

    TEST(Uniform_cumulants)
    {
        const double eps = 1.0e-10;

        double cum[2] = {3.9, 2.88};
        cumulant_test<UniformDistribution>(cum, sizeof(cum)/sizeof(cum[0]), eps);
    }

    TEST(Exponential_cumulants)
    {
        const double eps = 1.0e-10;

        double cum[2] = {3.9, 2.88};
        cumulant_test<ExponentialDistribution>(cum, sizeof(cum)/sizeof(cum[0]), eps);
    }

    TEST(LogNormal_st2)
    {
        for (unsigned i=0; i<100; ++i)
        {
            const double mean = test_rng() - 0.5;
            const double stdev = test_rng() + 0.5;
            const double skew = test_rng() - 0.5;
            const LogNormal lnd(mean, stdev, skew);
            standard_test_2(lnd, 1.e-6);
        }
    }

    TEST(LogNormal_cumulants)
    {
        const double eps = 1.0e-10;

        double cum[3] = {-1.0, 0.81, -1.458};
        cumulant_test<LogNormal>(cum, sizeof(cum)/sizeof(cum[0]), eps);
    }

    TEST(LogNormal_entropy)
    {
        const LogNormal d(1.23, 0.83, 1.0/7);
        const double xmin = d.quantile(0.0);
        const double xmax = d.quantile(1.0);
        entropy_test(d, xmin, xmax, 1024, 1, 1.0e-12);
    }

    TEST(LogNormal_median_sigmas)
    {
        const double eps = 1.0e-8;

        for (unsigned i=0; i<100; ++i)
        {
            const double median = test_rng() - 0.5;
            const double sigPlus = test_rng() + 0.1;
            const double sigMinus = test_rng() + 0.1;
            median_sigmas_test<LogNormal>(median, sigPlus, sigMinus, eps);
        }
    }

    TEST(JohnsonSu_entropy)
    {
        const JohnsonSu su(0.7, 0.9, 0.5, 5.0);
        CHECK(su.isValid());
        const LogJohnsonSuInverseTransform tr(su);
        const GaussHermiteQuadrature ghq(256);
        const double numEst = -ghq.integrateProb(0.0, 1.0, tr);
        CHECK_CLOSE(numEst, su.entropy(), 1.0e-12);
    }

    TEST(JohnsonSb_entropy)
    {
        const JohnsonSb sb(0.7, 0.9, 0.5, 3.0);
        CHECK(sb.isValid());
        const double xmin = sb.quantile(0);
        const double xmax = sb.quantile(1);
        entropy_test(sb, xmin, xmax, 1024, 1, 1.0e-12);
    }

    TEST(JohnsonSystem_cumulants)
    {
        const double eps = 1.0e-10;

        double cum1[] = {0.1, 1.2, 1.3, 17.0};
        cumulant_test<JohnsonSystem>(cum1, sizeof(cum1)/sizeof(cum1[0]), eps);

        double cum2[] = {0.1, 1.2, 1.3, 0.3};
        cumulant_test<JohnsonSystem>(cum2, sizeof(cum2)/sizeof(cum2[0]), eps);
    }

    TEST(JohnsonSystem_median_sigmas)
    {
        const double eps = 1.0e-8;

        for (unsigned i=0; i<200; ++i)
        {
            const double median = test_rng() - 0.5;
            const double sigPlus = test_rng() + 1.0;
            const double sigMinus = test_rng() + 1.0;
            median_sigmas_test<JohnsonSystem>(median, sigPlus, sigMinus, eps);
        }
    }

    TEST(InterpolatedDensity1D_1)
    {
        // In this test, InterpolatedDensity1D mimics the Gaussian
        const int nSigmas = 15;
        const double eps = 1.0e-8;
        const double eps2 = 1.0e-5;

        const Gaussian g(0.3, 1.1);
        const InterpolatedDensity1D distro(
            -nSigmas, nSigmas, 2*nSigmas*100 + 1,
            DensityFunctor1D(g), GaussianDensityDerivative(g));

        for (unsigned i=0; i<5; ++i)
            CHECK_CLOSE(g.cumulant(i), distro.cumulant(i), eps);

        // cdf values at integer sigmas will be calculated
        // by direct Gauss-Legendre integration
        for (int i=0; i<nSigmas; ++i)
        {
            CHECK_CLOSE(g.density(i), distro.density(i), eps);
            CHECK_CLOSE(1.0, g.cdf(i)/distro.cdf(i), eps2);
            CHECK_CLOSE(1.0, g.exceedance(i)/distro.exceedance(i), eps2);
            CHECK_CLOSE(g.density(-i), distro.density(-i), eps);
            CHECK_CLOSE(1.0, g.cdf(-i)/distro.cdf(-i), eps2);
            CHECK_CLOSE(1.0, g.exceedance(-i)/distro.exceedance(-i), eps2);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double x = 10.0*(test_rng() - 0.5);
            CHECK_CLOSE(g.density(x), distro.density(x), eps);
            const double cdf = g.cdf(x);
            CHECK_CLOSE(cdf, distro.cdf(x), eps);
            CHECK_CLOSE(g.invExceedance(cdf), distro.invExceedance(cdf), eps);
        }

        standard_test(distro, 5.0, 1.e-7);
        standard_test_2(distro, 1.e-7);        
        entropy_test(distro, -10.0, 10.0, 4, 2000, eps);
    }

    TEST(TabulatedDensity1D_1)
    {
        // In this test, TabulatedDensity1D mimics the Gaussian
        const int nSigmas = 15;
        const double eps = 1.0e-6;
        const double eps2 = 1.0e-4;

        const Gaussian g(0.0, 1.0);
        const TabulatedDensity1D distro(
            -nSigmas, nSigmas, 2*nSigmas*1000 + 1, DensityFunctor1D(g));

        for (unsigned i=0; i<5; ++i)
            CHECK_CLOSE(g.cumulant(i), distro.cumulant(i), eps);

        // cdf values at integer sigmas will be calculated
        // by direct Gauss-Legendre integration
        for (int i=0; i<nSigmas; ++i)
        {
            CHECK_CLOSE(g.density(i), distro.density(i), eps);
            CHECK_CLOSE(1.0, g.cdf(i)/distro.cdf(i), eps2);
            CHECK_CLOSE(1.0, g.exceedance(i)/distro.exceedance(i), eps2);
            CHECK_CLOSE(g.density(-i), distro.density(-i), eps);
            CHECK_CLOSE(1.0, g.cdf(-i)/distro.cdf(-i), eps2);
            CHECK_CLOSE(1.0, g.exceedance(-i)/distro.exceedance(-i), eps2);
        }

        for (unsigned i=0; i<100; ++i)
        {
            const double x = 10.0*(test_rng() - 0.5);
            CHECK_CLOSE(g.density(x), distro.density(x), eps);
            const double cdf = g.cdf(x);
            CHECK_CLOSE(cdf, distro.cdf(x), eps);
            CHECK_CLOSE(g.invExceedance(cdf), distro.invExceedance(cdf), eps);
        }

        standard_test(distro, 5.0, 1.e-7, false);
        standard_test_2(distro, 1.e-7);        
        entropy_test(distro, -10.0, 10.0, 4, 2000, eps);
    }

    TEST(TabulatedDensity1D_2)
    {
        const double eps = 1.0e-12;

        const std::vector<double> values = {0.5, 1.0};
        const TabulatedDensity1D distro(-1.0, 1.0, values);
        standard_test(distro, 1.0, eps, true, false);
        standard_test_2(distro, eps);        
    }

    TEST(TabulatedDensity1D_3)
    {
        const double eps = 1.0e-12;

        const std::vector<double> values = {0.2, 1.0, 0.1};
        const TabulatedDensity1D distro(-1.0, 1.0, values);
        standard_test(distro, 1.0, eps, false);
        standard_test_2(distro, eps);        
    }

    TEST(InterpolatedDensity1D_isUnimodal)
    {
        const double kmin = 1.8;
        const double kmax = 1.9;
        const unsigned nK = 101;
        const double kstep = (kmax - kmin)/(nK - 1U);
        const unsigned nScan = 1001;
        const double shrink = 1.0 - 1.0e-6;
        std::vector<double> values(nScan);
        std::vector<double> derivs(nScan);

        bool haveUnimodals = false, haveBimodals = false;
        for (unsigned ik=0; ik<nK; ++ik)
        {
            const double k = kmin + kstep*ik;
            const JohnsonSb j(0.0, 1.0, 0.0, k);
            if (j.isUnimodal())
                haveUnimodals = true;
            else
                haveBimodals = true;
            const double xmin = j.quantile(0.0)*shrink;
            const double xmax = j.quantile(1.0)*shrink;
            const EquidistantGrid grid(nScan, xmin, xmax);
            for (unsigned iscan=0; iscan<nScan; ++iscan)
            {
                const double x = grid.coordinate(iscan);
                values[iscan] = j.density(x);
                derivs[iscan] = j.densityDerivative(x);
            }
            const InterpolatedDensity1D intd(xmin, xmax, values, derivs);
            CHECK_EQUAL(j.isUnimodal(), intd.isUnimodal());
        }
        assert(haveUnimodals);
        assert(haveBimodals);
    }

    TEST(TabulatedDensity1D_isUnimodal)
    {
        const double kmin = 1.8;
        const double kmax = 1.9;
        const unsigned nK = 101;
        const double kstep = (kmax - kmin)/(nK - 1U);
        const unsigned nScan = 4001;
        const double shrink = 1.0 - 1.0e-6;
        std::vector<double> values(nScan);

        bool haveUnimodals = false, haveBimodals = false;
        for (unsigned ik=0; ik<nK; ++ik)
        {
            const double k = kmin + kstep*ik;
            const JohnsonSb j(0.0, 1.0, 0.0, k);
            if (j.isUnimodal())
                haveUnimodals = true;
            else
                haveBimodals = true;
            const double xmin = j.quantile(0.0)*shrink;
            const double xmax = j.quantile(1.0)*shrink;
            const EquidistantGrid grid(nScan, xmin, xmax);
            for (unsigned iscan=0; iscan<nScan; ++iscan)
            {
                const double x = grid.coordinate(iscan);
                values[iscan] = j.density(x);
            }
            const TabulatedDensity1D intd(xmin, xmax, values);
            CHECK_EQUAL(j.isUnimodal(), intd.isUnimodal());
        }
        assert(haveUnimodals);
        assert(haveBimodals);
    }

    TEST(LegendreDistro1D_1)
    {
        const double eps = 1.0e-10;

        const std::vector<double> coeffs{0.2, 0.13, 0.07};
        const LegendreDistro1D distro(0.0, 1.5, coeffs);

        CHECK(coeffs.size() == distro.nCoeffs());
        for (unsigned i=0; i<coeffs.size(); ++i)
            CHECK(coeffs[i] == distro.getCoeff(i));
        standard_test(distro, 1.5, eps, true, false);
        standard_test_2(distro, eps);

        CHECK(1.0 == distro.cumulant(0));
        CHECK_CLOSE(0.3464101615137755, distro.cumulant(1), eps);
        CHECK_CLOSE(0.8044133022449836, distro.cumulant(2), eps);
        CHECK_CLOSE(-0.338451041802855, distro.cumulant(3), eps);
        CHECK_CLOSE(-0.716979828408228, distro.cumulant(4), eps);
    }

    TEST(LegendreDistro1D_2)
    {
        const double eps = 1.0e-10;

        const std::vector<double> coeffs{-0.17, 0.03, 0.05};
        const LegendreDistro1D distro(0.0, 1.5, coeffs);

        CHECK(coeffs.size() == distro.nCoeffs());
        for (unsigned i=0; i<coeffs.size(); ++i)
            CHECK(coeffs[i] == distro.getCoeff(i));
        standard_test(distro, 1.5, eps, true, false);
        standard_test_2(distro, eps);

        std::vector<long double> cums{
            1.0L, -0.294448637286709L, 0.7035492235949962L,
            0.3005255708269526L, -0.414366844330764L};
        for (unsigned i=0; i<5; ++i)
            CHECK_CLOSE((double)(cums[i]), distro.cumulant(i), eps);

        const std::vector<long double> expMoms{
            -0.294448637286709L, 0.7902492235949962L,
            -0.34648045648081166L, 1.0901235026474927L};
        cumulantsToMoments(&cums[1], &cums[1], 4U);
        for (unsigned i=0; i<4; ++i)
            CHECK_CLOSE(expMoms[i], cums[i+1], eps);
    }

    TEST(MixtureModel1D_1)
    {
        const double eps = 1.0e-10;

        const std::vector<double> coeffs3{-0.17, 0.03, 0.05};
        const LegendreDistro1D distro3(0.0, 1.5, coeffs3);

        for (unsigned itry=0; itry<3; ++itry)
        {
            const double w1 = test_rng() + 0.1;
            MixtureModel1D mix;
            mix.add(distro3, w1);
            standard_test(mix, 1.5, eps, true, false);
            standard_test_2(mix, eps);

            for (unsigned ii=0; ii<10; ++ii)
            {
                const double x = 3.0*(test_rng()-0.5);
                CHECK_CLOSE(distro3.density(x), mix.density(x), eps);
                CHECK_CLOSE(distro3.densityDerivative(x), mix.densityDerivative(x), eps);
                CHECK_CLOSE(distro3.cdf(x), mix.cdf(x), eps);
                CHECK_CLOSE(distro3.exceedance(x), mix.exceedance(x), eps);

                const double y = test_rng();
                CHECK_CLOSE(distro3.quantile(y), mix.quantile(y), eps);
                CHECK_CLOSE(distro3.invExceedance(y), mix.invExceedance(y), eps);
            }

            CHECK_CLOSE(distro3.cumulant(0), mix.cumulant(0), eps);
            CHECK_CLOSE(distro3.cumulant(1), mix.cumulant(1), eps);
            CHECK_CLOSE(distro3.cumulant(2), mix.cumulant(2), eps);
            CHECK_CLOSE(distro3.cumulant(3), mix.cumulant(3), eps);
            CHECK_CLOSE(distro3.cumulant(4), mix.cumulant(4), eps);
        }
    }

    TEST(MixtureModel1D_2)
    {
        const double eps = 1.0e-10;

        const std::vector<double> coeffs1{0.2, 0.13, 0.07};
        const LegendreDistro1D distro1(0.0, 1.5, coeffs1);
        const std::vector<double> coeffs2{-0.17, 0.03, 0.05};
        const LegendreDistro1D distro2(0.0, 1.5, coeffs2);

        for (unsigned itry=0; itry<10; ++itry)
        {
            const double w1 = test_rng();
            MixtureModel1D mix;
            mix.add(distro1, w1);
            mix.add(distro2, 1.0 - w1);
            standard_test(mix, 1.5, eps, true, false);
            standard_test_2(mix, eps);

            std::vector<double> coeffs3(3);
            for (unsigned i=0; i<3; ++i)
                coeffs3[i] = w1*coeffs1[i] + (1.0 - w1)*coeffs2[i];
            const LegendreDistro1D distro3(0.0, 1.5, coeffs3);

            for (unsigned ii=0; ii<10; ++ii)
            {
                const double x = 3.0*(test_rng()-0.5);
                CHECK_CLOSE(distro3.density(x), mix.density(x), eps);
                CHECK_CLOSE(distro3.densityDerivative(x), mix.densityDerivative(x), eps);
                CHECK_CLOSE(distro3.cdf(x), mix.cdf(x), eps);
                CHECK_CLOSE(distro3.exceedance(x), mix.exceedance(x), eps);

                const double y = test_rng();
                CHECK_CLOSE(distro3.quantile(y), mix.quantile(y), eps);
                CHECK_CLOSE(distro3.invExceedance(y), mix.invExceedance(y), eps);
            }

            CHECK_CLOSE(distro3.cumulant(0), mix.cumulant(0), eps);
            CHECK_CLOSE(distro3.cumulant(1), mix.cumulant(1), eps);
            CHECK_CLOSE(distro3.cumulant(2), mix.cumulant(2), eps);
            CHECK_CLOSE(distro3.cumulant(3), mix.cumulant(3), eps);
            CHECK_CLOSE(distro3.cumulant(4), mix.cumulant(4), eps);
        }
    }
}
