#include <cmath>
#include <cassert>

#include "UnitTest++.h"
#include "test_utils.hh"

#include "ase/LogLikelihoodCurves.hh"
#include "ase/LikelihoodCurveCopy.hh"
#include "ase/GeneralisedPoissonHelper.hh"
#include "ase/DistributionModels1D.hh"

using namespace ase;
using namespace std;

namespace {
    class TestDoubleCubicLogSigma : public DoubleCubicLogSigma
    {
    public:
        inline TestDoubleCubicLogSigma(const double location,
                                       const double sigmaPlus,
                                       const double sigmaMinus)
            : DoubleCubicLogSigma(location, sigmaPlus, sigmaMinus,
                                  hypot(sigmaPlus, sigmaMinus)/M_SQRT2) {}

        inline virtual ~TestDoubleCubicLogSigma() override {}

        inline virtual TestDoubleCubicLogSigma* clone() const override
            {return new TestDoubleCubicLogSigma(*this);}

        inline virtual std::string classname() const override
            {return "TestDoubleCubicLogSigma";}
    };

    double numericalDerivative(const AbsLogLikelihoodCurve& c,
                               const double x, const double h)
    {
        assert(h > 0.0);
        const volatile double xplus = x + h;
        const volatile double xminus = x - h;
        return (c(xplus) - c(xminus))/(xplus - xminus);
    }

    double numericalSecondDerivative(const AbsLogLikelihoodCurve& c,
                                     const double x)
    {
        const double h = 0.02*c.stepSize();
        const volatile double xplus = x + h;
        const volatile double xminus = x - h;
        return (c.derivative(xplus) - c.derivative(xminus))/(xplus - xminus);
    }

    template<class Curve>
    void standard_test(const double location,
                       const double sigmaPlus, const double sigmaMinus,
                       const double range, const double eps,
                       const bool testSecondDerivative = true)
    {
        assert(sigmaPlus > 0.0);
        assert(sigmaMinus > 0.0);
        const Curve c(location, sigmaPlus, sigmaMinus);
        // c.posteriorMean();
        // c.posteriorVariance();
        const double factor = test_rng() + 0.5;
        const auto& cprod = factor*c;
        const double loc = c.location();
        CHECK_EQUAL(location, loc);
        const double peak = c.argmax();
        CHECK_EQUAL(peak, loc);
        CHECK_CLOSE(0.0, c.derivative(peak), eps);
        const double sigP = c.sigmaPlus();
        CHECK(sigP > 0.0);
        CHECK_CLOSE(sigP, sigmaPlus, eps);
        const double sigM = c.sigmaMinus();
        CHECK(sigM > 0.0);
        CHECK_CLOSE(sigM, sigmaMinus, eps);
        const double logliMax = c.maximum();
        CHECK_CLOSE(0.5, logliMax-c(location+sigmaPlus), eps);
        CHECK_CLOSE(0.5, logliMax-c(location-sigmaMinus), eps);
        double xmin = peak - range*sigmaMinus;
        if (xmin < c.parMin() + c.stepSize())
            xmin = c.parMin() + c.stepSize();
        double xmax = peak + range*sigmaPlus;
        if (xmax > c.parMax() - c.stepSize())
            xmax = c.parMax() - c.stepSize();
        const double h = 0.01*c.stepSize();
        for (unsigned i=0; i<100; ++i)
        {
            const double x = xmin + test_rng()*(xmax - xmin);
            const double nd = numericalDerivative(c, x, h);
            const double deriv = c.derivative(x);
            CHECK_CLOSE(nd, deriv, eps*(fabs(deriv) + 1.0));

            if (testSecondDerivative)
            {
                const double nd2 = numericalSecondDerivative(c, x);
                const double deriv2 = c.secondDerivative(x);
                CHECK_CLOSE(nd2, deriv2, 20.0*eps*(fabs(deriv2) + 1.0));
            }

            const double logli = c(x);
            const double deltaLogli = logliMax - logli;
            if (x > peak)
            {
                const double del = x - peak;
                const double s = c.sigmaPlus(deltaLogli);
                CHECK_CLOSE(del, s, eps);
            }
            else
            {
                const double del = peak - x;
                const double s = c.sigmaMinus(deltaLogli);
                CHECK_CLOSE(del, s, eps);
            }

            CHECK_CLOSE(deriv*factor, cprod.derivative(x), 1.0e-15*std::abs(deriv*factor));
            CHECK_CLOSE(logli*factor, cprod(x), 1.0e-15*std::abs(logli*factor));
        }
    }

    TEST(SymmetrizedParabola_1)
    {
        const double eps = 1.0e-11;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig = test_rng() + 0.1;
            standard_test<SymmetrizedParabola>(mu, sig, sig, range, eps);
            const SymmetrizedParabola sp(mu, sig, sig);
            CHECK_CLOSE(mu, sp.posteriorMean(), eps);
            CHECK_CLOSE(sig*sig, sp.posteriorVariance(), eps);
        }
    }

    TEST(BrokenParabola_1)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;

            standard_test<BrokenParabola>(mu, sig1, sig1, range, eps);

            // Turn off the second derivative test for different sigmas
            // because dimidated parabola has discontinuous second derivative
            standard_test<BrokenParabola>(mu, sig1, sig2, range, eps, false);

            const BrokenParabola bp(mu, sig1, sig2);
            const FechnerDistribution fech(mu, sig1, sig2);
            CHECK_CLOSE(fech.cumulant(1), bp.posteriorMean(), 1.0e-12);
            CHECK_CLOSE(fech.cumulant(2), bp.posteriorVariance(), 1.0e-12);
        }
    }

    TEST(LogarithmicLogli_1)
    {
        const double eps = 1.0e-3;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;
            standard_test<LogarithmicLogli>(mu, sig1, sig1, range, eps);
            standard_test<LogarithmicLogli>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(TruncatedCubicLogli_1)
    {
        const double eps = 1.0e-7;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<TruncatedCubicLogli>(mu, sig1, sig1, range, eps);
            standard_test<TruncatedCubicLogli>(mu, sig1, sig2, range, eps);
            const TruncatedCubicLogli tcl(mu, sig1, sig2);
            tcl.posteriorMean();
            tcl.posteriorVariance();
        }
    }

    TEST(GeneralisedPoissonHelper_1)
    {
        const double eps = 1.0e-3;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.2;
            const double sig2 = test_rng() + 0.2;
            standard_test<GeneralisedPoissonHelper>(mu, sig1, sig1, range, eps);
            if (sig1 > sig2)
                standard_test<GeneralisedPoissonHelper>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(GeneralisedPoisson_1)
    {
        const double eps = 1.0e-3;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;
            standard_test<GeneralisedPoisson>(mu, sig1, sig1, range, eps);
            standard_test<GeneralisedPoisson>(mu, sig1, sig2, range, eps);
            const GeneralisedPoisson gp(mu, sig1, sig2);
            const double mean1 = gp.posteriorMean();
            gp.posteriorVariance();
            const double shift = test_rng();
            const GeneralisedPoisson gp2(mu + shift, sig1, sig2);
            const double mean2 = gp2.posteriorMean();
            CHECK_CLOSE(shift, mean2-mean1, 1.0e-12);
        }
    }

    TEST(ConstrainedQuartic_1)
    {
        const double eps = 1.0e-8;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<ConstrainedQuartic>(mu, sig1, sig1, range, eps);
            standard_test<ConstrainedQuartic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(MoldedQuartic_1)
    {
        const double eps = 1.0e-8;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<MoldedQuartic>(mu, sig1, sig1, range, eps);
            standard_test<MoldedQuartic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(DoubleCubicLogSigma_1)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<TestDoubleCubicLogSigma>(mu, sig1, sig1, range, eps);
            standard_test<TestDoubleCubicLogSigma>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(MoldedCubicLogSigma_1)
    {
        const double eps = 1.0e-5;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<MoldedCubicLogSigma>(mu, sig1, sig1, range, eps);
            standard_test<MoldedCubicLogSigma>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(QuinticLogSigma_1)
    {
        const double eps = 1.0e-7;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<QuinticLogSigma>(mu, sig1, sig1, range, eps);
            standard_test<QuinticLogSigma>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(LogLogisticBeta_1)
    {
        const double eps = 1.0e-7;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<LogLogisticBeta>(mu, sig1, sig1, range, eps);
            standard_test<LogLogisticBeta>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(VariableLogSigma_1)
    {
        const double eps = 1.0e-8;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<VariableLogSigma>(mu, sig1, sig1, range, eps);
            standard_test<VariableLogSigma>(mu, sig1, sig2, range, eps);
        }
    }

    /* Disable for now due to discontinuous derivative
    TEST(PDGLogli_1)
    {
        const double eps = 1.0e-8;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<PDGLogli>(mu, sig1, sig1, range, eps);
            standard_test<PDGLogli>(mu, sig1, sig2, range, eps);
        }
    }
    */

    TEST(MatchedQuintic_1)
    {
        const double eps = 1.0e-8;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            const MatchedQuintic q(mu, sig1, sig2);

            double p = sig1 + eps;
            double m = sig1 - eps;
            CHECK_CLOSE(q(p), q(m), 10*eps);
            CHECK_CLOSE(q.derivative(p), q.derivative(m), 10*eps);
            CHECK_CLOSE(q.secondDerivative(p), q.secondDerivative(m), 10*eps);

            p = -sig2 + eps;
            m = -sig2 - eps;
            CHECK_CLOSE(q(p), q(m), 10*eps);
            CHECK_CLOSE(q.derivative(p), q.derivative(m), 10*eps);
            CHECK_CLOSE(q.secondDerivative(p), q.secondDerivative(m), 10*eps);
        }
    }

    TEST(MatchedQuintic_2)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<MatchedQuintic>(mu, sig1, sig1, range, eps);
            standard_test<MatchedQuintic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(Interpolated7thDegree_1)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<Interpolated7thDegree>(mu, sig1, sig1, range, eps);
            standard_test<Interpolated7thDegree>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(MoldedDoubleQuartic_1)
    {
        const double eps = 1.0e-5;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<MoldedDoubleQuartic>(mu, sig1, sig1, range, eps);
            standard_test<MoldedDoubleQuartic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(SimpleDoubleQuartic_1)
    {
        const double eps = 1.0e-5;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<SimpleDoubleQuartic>(mu, sig1, sig1, range, eps);
            standard_test<SimpleDoubleQuartic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(MoldedDoubleQuintic_1)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<MoldedDoubleQuintic>(mu, sig1, sig1, range, eps);
            standard_test<MoldedDoubleQuintic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(SimpleDoubleQuintic_1)
    {
        const double eps = 1.0e-6;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.5;
            const double sig2 = test_rng() + 0.5;
            standard_test<SimpleDoubleQuintic>(mu, sig1, sig1, range, eps);
            standard_test<SimpleDoubleQuintic>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(VariableSigmaLogli_1)
    {
        const double eps = 1.0e-3;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;
            standard_test<VariableSigmaLogli>(mu, sig1, sig1, range, eps);
            standard_test<VariableSigmaLogli>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(VariableVarianceLogli_1)
    {
        const double eps = 1.0e-3;
        const double range = 5.0;

        for (unsigned i=0; i<20U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 0.1;
            const double sig2 = test_rng() + 0.1;
            standard_test<VariableVarianceLogli>(mu, sig1, sig1, range, eps);
            standard_test<VariableVarianceLogli>(mu, sig1, sig2, range, eps);
        }
    }

    TEST(ConservativeSigma_1)
    {
        const double eps = 1.0e-5;
        const double range = 5.0;

        for (unsigned i=0; i<100U; ++i)
        {
            const double mu = test_rng();
            const double sig1 = test_rng() + 1.0;
            const double sig2 = test_rng() + 1.0;
            standard_test<ConservativeSigma05>(mu, sig1, sig1, range, eps);
            standard_test<ConservativeSigma05>(mu, sig1, sig2, range, eps);
            standard_test<ConservativeSigma10>(mu, sig1, sig1, range, eps);
            standard_test<ConservativeSigma10>(mu, sig1, sig2, range, eps);
            standard_test<ConservativeSigma15>(mu, sig1, sig1, range, eps);
            standard_test<ConservativeSigma15>(mu, sig1, sig2, range, eps);
            standard_test<ConservativeSigma20>(mu, sig1, sig1, range, eps);
            standard_test<ConservativeSigma20>(mu, sig1, sig2, range, eps);
            standard_test<ConservativeSigmaMax>(mu, sig1, sig1, range, eps);
            standard_test<ConservativeSigmaMax>(mu, sig1, sig2, range, eps);
            const ConservativeSigma10 cs10(mu, sig1, sig2);
            cs10.posteriorMean();
            cs10.posteriorVariance();
            const ConservativeSigmaMax csm(mu, sig1, sig2);
            csm.posteriorMean();
            csm.posteriorVariance();
        }
    }
}
