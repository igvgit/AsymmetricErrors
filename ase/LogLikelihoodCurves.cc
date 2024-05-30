#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <climits>

#include "ase/mathUtils.hh"
#include "ase/gCdfValues.hh"
#include "ase/DistributionModels1D.hh"
#include "ase/LogLikelihoodCurves.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/findMinimumGoldenSection.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/PosteriorMomentFunctor.hh"

namespace {
    class Fcn1
    {
    public:
        inline Fcn1(const double sp, const double sm)
            : sp_(sp), sm_(sm) {}

        inline double operator()(const double c) const
        {
            assert(c > 0.0);
            return fcn(sp_, c) - fcn(-sm_, c);
        }

    private:
        inline double fcn(const double sigma, const double c) const
            {return exp(-sigma/c) + sigma/c;}

        double sp_;
        double sm_;
    };

    double llc_solveEq1(const double sp, const double sm)
    {
        assert(sp > 0.0);
        assert(sm > 0.0);
        assert(sp > sm);

        const Fcn1 fcn(sp, sm);
        const double cmin = sm/log(DBL_MAX/2.0);
        const double fcnMin = fcn(cmin);

        double c = cmin;
        for (unsigned i=0; i<2000; ++i)
        {
            c *= 2.0;
            if (fcn(c)*fcnMin < 0.0)
            {
                const double status = ase::findRootUsingBisections(
                    fcn, 0.0, cmin, c, 2.0*DBL_EPSILON, &c);
                assert(status);
                return c;
            }
        }
        std::ostringstream os;
        os.precision(16);
        os << "In llc_solveEq1: failed to find a good starting point "
           << "for sp = " << sp << ", sm = " << sm;
        throw std::runtime_error(os.str());
        return 0.0;
    }

    class Fcn2
    {
    public:
        inline Fcn2(const double sp, const double sm, const double c)
            : sp_(sp), sm_(sm), c_(c) {}

        inline double operator()(const double a) const
        {
            assert(std::abs(a) < 1.0);
            return fcn(sp_, a, c_) - fcn(-sm_, a, c_);
        }

    private:
        inline double fcn(const double x, const double a,
                          const double c) const
        {
            const double tmp = exp(x/c);
            const double tmp2 = (1 + tmp + a*(tmp - 1))/2;
            return (a - 1)*log(tmp2) - (a + 1)*log(tmp2/tmp);
        }

        double sp_;
        double sm_;
        double c_;
    };

    // For the given value of parameter "c", this function finds
    // the value of parameter "a" such that ln L(sp) == ln L(-sm)
    double llc_solveEq2(const double sp, const double sm, const double c)
    {
        assert(sp > 0.0);
        assert(sm > 0.0);
        assert(sp > sm);
        assert(c > 0.0);

        const Fcn2 fcn(sp, sm, c);
        double step = 1.0;

        // a = 1.0 will be a solution. But we need another one...
        const double fcn0 = fcn(0.0);
        for (unsigned i=0; i<100; ++i)
        {
            step /= 2.0;
            const double atry = 1.0 - step;
            if (atry == 1.0)
               break;
            if (fcn(atry)*fcn0 < 0.0)
            {
                double a;
                const double status = ase::findRootUsingBisections(
                    fcn, 0.0, 0.0, atry, 2.0*DBL_EPSILON, &a);
                assert(status);
                return a;
            }
        }
        std::ostringstream os;
        os.precision(17);
        os << "In llc_solveEq2: failed to find a good starting point "
           << "for sp = " << sp << ", sm = " << sm << ", c = " << c;
        throw std::runtime_error(os.str());
        return 0.0;
    }

    double llc_logLogisticBeta(const double x, const double a,
                               const double c, const double g)
    {
        assert(c > 0.0);
        assert(std::abs(a) < 1.0);
        assert(g > 0.0);

        const double tmp = exp(x/c);
        const double tmp2 = (1 + tmp + a*(tmp - 1))/2;
        return g*c*c*((a - 1)*log(tmp2) - (a + 1)*log(tmp2/tmp));
    }

    double llc_reducedLogisticBeta(const double sp, const double sm,
                                   const double c)
    {
        const double a = llc_solveEq2(sp, sm, c);
        return llc_logLogisticBeta(sp, a, c, 1.0);
    }

    class Fcn3
    {
    public:
        inline Fcn3(const double sp, const double sm)
            : sp_(sp), sm_(sm) {}

        inline double operator()(const double c) const
            {return llc_reducedLogisticBeta(sp_, sm_, c);}

    private:
        double sp_;
        double sm_;
    };

    std::pair<double,double> llc_optimizeLogisticBeta(
        const double sp, const double sm)
    {
        const double tol = sqrt(DBL_EPSILON);

        // If you are getting exceptions like "In llc_solveEq2: failed
        // to find a good starting point for sp = ..." you might need
        // to increase the offset defined below.
        const double offset = 20.0*tol;

        const Fcn3 fcn(sp, sm);
        double cmax = (1.0 - offset)*llc_solveEq1(sp, sm);
        double fmax = fcn(cmax);
        double cmed = cmax/2.0;
        double fmed = fcn(cmed);
        double cmin = cmed/2.0;
        double fmin = fcn(cmin);
        for (unsigned i=0; i<100; ++i)
        {
            if (ase::isMinimumBracketed(fmin, fmed, fmax))
            {
                double c;
                const bool status = ase::findMinimumGoldenSection(
                    fcn, cmin, cmed, cmax, tol, &c);
                assert(status);
                const double a = llc_solveEq2(sp, sm, c);
                return std::make_pair(a, c);
            }
            cmax = cmed;
            fmax = fmed;
            cmed = cmin;
            fmed = fmin;
            cmin = cmed/2.0;
            fmin = fcn(cmin);
        }
        std::ostringstream os;
        os.precision(16);
        os << "In llc_optimizeLogisticBeta: failed to find a good starting point "
           << "for sp = " << sp << ", sm = " << sm;
        throw std::runtime_error(os.str());
        return std::make_pair(0.0, 0.0);
    }

    double conservativeSplineConstructorArgument(
        const double sigmaPlus, const double sigmaMinus,
        const double sigmaTarget)
    {
        const double vtarget = sigmaTarget*sigmaTarget;
        const double maxtarget = ase::ConservativeSpline::maxDerivLimitFactor(
            sigmaPlus, sigmaMinus);
        return std::min(vtarget, maxtarget);
    }

    class ConcreteDoubleCubicLogSigma : public ase::DoubleCubicLogSigma
    {
        using ase::DoubleCubicLogSigma::DoubleCubicLogSigma;

        inline virtual ConcreteDoubleCubicLogSigma* clone() const override
            {return new ConcreteDoubleCubicLogSigma(*this);}

        inline virtual std::string classname() const override
            {return "ConcreteDoubleCubicLogSigma";}
    };

    class LogliDeltaSquared
    {
    public:
        inline LogliDeltaSquared(const ase::AbsLogLikelihoodCurve& c1,
                                 const ase::AbsLogLikelihoodCurve& c2)
            : c1_(c1), c2_(c2) {}

        inline double operator()(const double parameter) const
        {
            const double del = c1_(parameter) - c2_(parameter);
            return del*del;
        }

    private:
        const ase::AbsLogLikelihoodCurve& c1_;
        const ase::AbsLogLikelihoodCurve& c2_;
    };

    double logliDeltaSquaredIntegral(const ase::AbsLogLikelihoodCurve& c1,
                                     const ase::AbsLogLikelihoodCurve& c2,
                                     const double xmin, const double xmax,
                                     const unsigned npoints)
    {
        assert(xmin < xmax);
        const LogliDeltaSquared fcn(c1, c2);
        const ase::GaussLegendreQuadrature glq(npoints);
        return glq.integrate(fcn, xmin, xmax);
    }

    class MoldedCubicLogSigmaOpt
    {
    public:
        inline MoldedCubicLogSigmaOpt(const double sp, const double sm,
                                      const unsigned npoints)
            : sp_(sp), sm_(sm), ref_(0.0, sp, sm), npoints_(npoints) {}

        inline double operator()(const double sig0) const
        {
            const ConcreteDoubleCubicLogSigma c(0.0, sp_, sm_, sig0);
            const double leftI = logliDeltaSquaredIntegral(c, ref_, -sm_, 0.0, npoints_);
            const double rightI = logliDeltaSquaredIntegral(c, ref_, 0.0, sp_, npoints_);
            return leftI/sm_ + rightI/sp_;
        }

    private:
        double sp_;
        double sm_;
        ase::BrokenParabola ref_;
        unsigned npoints_;
    };
}

namespace ase {
    // Initialize static members
    const double DistributionLogli::defaultDensityCutoff_ = sqrt(DBL_MIN);
    const double ConservativeSpline::tol_ = 32.0*DBL_EPSILON;

    /********************************************************************/

    BrokenParabola::BrokenParabola(const double i_location,
                                         const double i_sigmaPlus,
                                         const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::BrokenParabola constructor",
                       sigPlus_, sigMinus_);
    }

    double BrokenParabola::stepSize() const
    {
        return 0.01*std::min(sigPlus_,sigMinus_);
    }

    double BrokenParabola::uValue(const double x) const
    {
        const double s = x >= 0.0 ? sigPlus_ : sigMinus_;
        const double del = x/s;
        return -del*del/2.0;
    }

    double BrokenParabola::uDerivative(const double x) const
    {
        const double s = x >= 0.0 ? sigPlus_ : sigMinus_;
        return -x/s/s;
    }

    double BrokenParabola::uSecondDerivative(const double x, double) const
    {
        const double s = x >= 0.0 ? sigPlus_ : sigMinus_;
        return -1.0/s/s;
    }

    double BrokenParabola::uSigmaPlus(const double deltaLogLikelihood,
                                         double) const
    {
        return sqrt(2.0*deltaLogLikelihood)*sigPlus_;
    }

    double BrokenParabola::uSigmaMinus(const double deltaLogLikelihood,
                                          double) const
    {
        return sqrt(2.0*deltaLogLikelihood)*sigMinus_;
    }

    /*********************************************************************/

    SymmetrizedParabola::SymmetrizedParabola(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : AbsShiftableLogli(i_location)
    {
        validateSigmas("ase::SymmetrizedParabola constructor",
                       i_sigmaPlus, i_sigmaMinus);
        if (i_sigmaPlus == i_sigmaMinus)
            sigPlus_ = i_sigmaPlus;
        else
        {
            const FechnerDistribution fd(i_location, i_sigmaPlus, i_sigmaMinus);
            setShift(fd.cumulant(1));
            sigPlus_ = sqrt(fd.cumulant(2));
        }
    }

    double SymmetrizedParabola::stepSize() const
    {
        return 0.01*sigPlus_;
    }

    double SymmetrizedParabola::uValue(const double x) const
    {
        const double del = x/sigPlus_;
        return -del*del/2.0;
    }

    double SymmetrizedParabola::uDerivative(const double x) const
    {
        return -x/sigPlus_/sigPlus_;
    }

    double SymmetrizedParabola::uSecondDerivative(double, double) const
    {
        return -1.0/sigPlus_/sigPlus_;
    }

    double SymmetrizedParabola::uSigmaPlus(const double deltaLogLikelihood,
                                           double) const
    {
        return sqrt(2.0*deltaLogLikelihood)*sigPlus_;
    }

    double SymmetrizedParabola::uSigmaMinus(const double deltaLogLikelihood,
                                            double) const
    {
        return sqrt(2.0*deltaLogLikelihood)*sigPlus_;
    }

    /*********************************************************************/

    LogarithmicLogli::LogarithmicLogli(const double i_location,
                                       const double i_sigmaPlus,
                                       const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::LogarithmicLogli constructor",
                       sigPlus_, sigMinus_);
        logbeta_ = log(sigPlus_/sigMinus_);
        gamma_ = (sigPlus_ - sigMinus_)/sigPlus_/sigMinus_;
        if (gamma_ == 0.0)
        {
            pmin_ = -DBL_MAX;
            pmax_ = DBL_MAX;
        }
        else if (gamma_ > 0.0)
        {
            pmin_ = -(1.0 - DBL_EPSILON)/gamma_;
            pmax_ = DBL_MAX;
        }
        else
        {
            pmin_ = -DBL_MAX;
            pmax_ = -(1.0 - DBL_EPSILON)/gamma_;
        }
    }

    double LogarithmicLogli::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double LogarithmicLogli::uValue(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::LogarithmicLogli::uValue: "
            "argument is out of range");
        if (gamma_ == 0.0)
        {
            const double del = x/sigPlus_;
            return -del*del/2.0;
        }
        else
        {
            const double del = log(1.0 + gamma_*x)/logbeta_;
            return -del*del/2.0;
        }
    }

    double LogarithmicLogli::uDerivative(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::LogarithmicLogli::uDerivative: "
            "argument is out of range");
        if (gamma_ == 0.0)
            return -x/sigPlus_/sigPlus_;
        else
        {
            const double tmp = 1.0 + gamma_*x;
            const double del = log(tmp)/logbeta_;
            return -del/logbeta_/tmp*gamma_;
        }
    }

    double LogarithmicLogli::uSecondDerivative(const double x, double) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::LogarithmicLogli::uSecondDerivative: "
            "argument is out of range");
        if (gamma_ == 0.0)
            return -1.0/sigPlus_/sigPlus_;
        else
        {
            const double tmp = logbeta_ + gamma_*logbeta_*x;
            return gamma_*gamma_*(log(1.0 + gamma_*x) - 1.0)/tmp/tmp;
        }
    }

    double LogarithmicLogli::uSigmaPlus(const double deltaLogLikelihood,
                                        double) const
    {
        if (gamma_ == 0.0)
            return sqrt(2.0*deltaLogLikelihood)*sigPlus_;
        else
        {
            const double tmp = sqrt(2.0*deltaLogLikelihood)*logbeta_;
            return (exp(tmp) - 1.0)/gamma_;
        }
    }

    double LogarithmicLogli::uSigmaMinus(const double deltaLogLikelihood,
                                         double) const
    {
        if (gamma_ == 0.0)
            return sqrt(2.0*deltaLogLikelihood)*sigPlus_;
        else
        {
            const double tmp = sqrt(2.0*deltaLogLikelihood)*logbeta_;
            return (1.0 - exp(-tmp))/gamma_;
        }
    }

    /*********************************************************************/

    TruncatedCubicLogli::TruncatedCubicLogli(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::TruncatedCubicLogli constructor",
                       i_sigmaPlus, i_sigmaMinus);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio < 0.5 || sigratio > 2.0) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli constructor: "
            "sigma asymmetry is too high");
        const double ssum = sigPlus_ + sigMinus_;
        const double sp2 = sigPlus_*sigPlus_;
        const double sm2 = sigMinus_*sigMinus_;
        const double sp3 = sp2*sigPlus_;
        const double sm3 = sm2*sigMinus_;
        alpha_ = (sp3 + sm3)/sp2/sm2/ssum;
        beta_ = (sm2 - sp2)/sp2/sm2/ssum;
        if (beta_ == 0.0)
        {
            pmin_ = -DBL_MAX;
            pmax_ = DBL_MAX;
            truncationLogli_ = -DBL_MAX;
        }
        else
        {
            const double xlim = -2.0*alpha_/3.0/beta_;
            if (xlim > 0)
            {
                pmin_ = -DBL_MAX;
                pmax_ = xlim;
            }
            else
            {
                pmin_ = xlim;
                pmax_ = DBL_MAX;
            }
            truncationLogli_ = uValue(xlim);
        }
    }

    double TruncatedCubicLogli::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double TruncatedCubicLogli::uValue(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli::uValue: "
            "argument is out of range");
        return -0.5*x*x*(alpha_ + beta_*x);
    }

    double TruncatedCubicLogli::uDerivative(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli::uDerivative: "
            "argument is out of range");
        return -0.5*x*(2.0*alpha_ + 3.0*beta_*x);
    }

    double TruncatedCubicLogli::uSecondDerivative(const double x, double) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli::uSecondDerivative: "
            "argument is out of range");
        return -alpha_ - 3.0*beta_*x;
    }

    double TruncatedCubicLogli::uSigmaPlus(const double deltaLogLikelihood,
                                           double) const
    {
        const double r = deltaLogLikelihood;
        if (beta_ == 0.0)
            return sqrt(2.0*r)*sigPlus_;
        if (beta_ < 0.0 && r >= -truncationLogli_) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli::uSigmaPlus: "
            "deltaLogLikelihood argument is too large");
        double roots[3];
        const unsigned nRoots = solveCubic(alpha_/beta_, 0.0, -2.0*r/beta_, roots);
        if (nRoots == 3U)
        {
            std::sort(roots, roots+3);
            if (beta_ < 0.0)
                return roots[1];
            else
                return roots[2];
        }
        else
        {
            assert(nRoots == 1U);
            return roots[0];
        }
    }

    double TruncatedCubicLogli::uSigmaMinus(const double deltaLogLikelihood,
                                            double) const
    {
        const double r = deltaLogLikelihood;
        if (beta_ == 0.0)
            return sqrt(2.0*r)*sigMinus_;
        if (beta_ > 0.0 && r >= -truncationLogli_) throw std::invalid_argument(
            "In ase::TruncatedCubicLogli::uSigmaMinus: "
            "deltaLogLikelihood argument is too large");
        double roots[3];
        const unsigned nRoots = solveCubic(alpha_/beta_, 0.0, -2.0*r/beta_, roots);
        if (nRoots == 3U)
        {
            std::sort(roots, roots+3);
            if (beta_ > 0.0)
                return -roots[1];
            else
                return -roots[0];
        }
        else
        {
            assert(nRoots == 1U);
            return -roots[0];
        }
    }

    /*********************************************************************/

    ConstrainedQuartic::ConstrainedQuartic(const double i_location,
                                           const double i_sigmaPlus,
                                           const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          beta_(0.0)
    {
        static const double maxRatio = (1.0 + M_SQRT2*pow(3.0, 0.25) + sqrt(3.0))/2.0;

        validateSigmas("ase::ConstrainedQuartic constructor",
                       sigPlus_, sigMinus_);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio >= maxRatio || 1.0/sigratio >= maxRatio)
            throw std::invalid_argument(
                "In ase::ConstrainedQuartic constructor: "
                "sigma asymmetry is too high");

        if (sigMinus_ == sigPlus_)
            alpha_ = M_SQRT2/sigPlus_;
        else
        {
            const double sm = sigMinus_;
            const double sm2 = sm*sm;
            const double sm3 = sm*sm2;
            const double sm4 = sm2*sm2;
            const double sp = sigPlus_;
            const double sp2 = sp*sp;
            const double sp3 = sp*sp2;
            const double sp4 = sp2*sp2;
            const double smplussp = sm + sp;

            const double tmp1 = 4*sm3*sp + 4*sm*sp3 - 2*sm4 - 2*sp4;
            assert(tmp1 >= 0.0);
            const double innerSqrt = sqrt(tmp1);
            const double tmp2 = 3*smplussp*smplussp;
            const double denom = 3*sm2 + 2*sm*sp + 3*sp2;
            const double tmp3 = (tmp2 + 6*innerSqrt)/denom;
            const double tmp4 = (tmp2 - 6*innerSqrt)/denom;
            const double absBeta = 2.0*(tmp4 < 0.0 ? sqrt(tmp3) : sqrt(tmp4))/sp/sm;
            beta_ = sp < sm ? absBeta : -absBeta;
            const double tmp5 = 72.0 - 2.0*beta_*beta_*sp4;
            assert(tmp5 >= 0.0);
            const double tmp6 = sqrt(tmp5)/6.0/sp;
            const double tmp7 = -beta_*sp/3.0;
            const double alpha1 = tmp7 + tmp6;
            const double alpha2 = tmp7 - tmp6;
            const double tmp8 = 72.0 - 2.0*beta_*beta_*sm4;
            assert(tmp8 >= 0.0);
            const double tmp9 = sqrt(tmp8)/6.0/sm;
            const double tmp10 = beta_*sm/3.0;
            const double alpha3 = tmp10 + tmp9;
            const double alpha4 = tmp10 - tmp9;
            const double match3 = std::min(std::abs(alpha3 - alpha1), std::abs(alpha3 - alpha2));
            const double match4 = std::min(std::abs(alpha4 - alpha1), std::abs(alpha4 - alpha2));
            if (match4 < match3)
            {
                if (std::abs(alpha4 - alpha1) < std::abs(alpha4 - alpha2))
                    alpha_ = (alpha4 + alpha1)/2.0;
                else
                    alpha_ = (alpha4 + alpha2)/2.0;
            }
            else
            {
                if (std::abs(alpha3 - alpha1) < std::abs(alpha3 - alpha2))
                    alpha_ = (alpha3 + alpha1)/2.0;
                else
                    alpha_ = (alpha3 + alpha2)/2.0;
            }
        }
    }

    double ConstrainedQuartic::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double ConstrainedQuartic::uValue(const double x) const
    {
        return -x*x*(alpha_*alpha_/2.0 + x*(alpha_*beta_/3.0 + beta_*beta_*x/12.0))/2.0;
    }

    double ConstrainedQuartic::uDerivative(const double x) const
    {
        return -x*(alpha_*alpha_ + x*(alpha_*beta_ + beta_*beta_*x/3.0))/2.0;
    }

    double ConstrainedQuartic::uSecondDerivative(const double x, double) const
    {
        const double tmp = alpha_ + beta_*x;
        return -tmp*tmp/2.0;
    }

    double ConstrainedQuartic::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double ConstrainedQuartic::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    MoldedQuartic::MoldedQuartic(const double i_location,
                                 const double i_sigmaPlus,
                                 const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          a_(0.0), b_(0.0), c_(0.0)
    {
        static const double maxRatio = 3.4080405968706883;

        validateSigmas("ase::MoldedQuartic constructor",
                       sigPlus_, sigMinus_);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio >= maxRatio || 1.0/sigratio >= maxRatio)
            throw std::invalid_argument(
                "In ase::MoldedQuartic constructor: "
                "sigma asymmetry is too high");

        if (sigMinus_ == sigPlus_)
            c_ = 1.0/sigPlus_/sigPlus_;
        else
        {
            const long double sm = sigMinus_;
            const long double sp = sigPlus_;

            const long double ssum = sm + sp;
            const long double ssum2 = ssum*ssum;
            const long double ssum4 = ssum2*ssum2;

            const long double sdiff = sm - sp;
            const long double sdiff2 = sdiff*sdiff;

            const long double sm2 = sm*sm;
            const long double sm3 = sm*sm2;
            const long double sm4 = sm2*sm2;
            const long double sm5 = sm2*sm3;
            const long double sm6 = sm3*sm3;
            const long double sm7 = sm3*sm4;
            const long double sm8 = sm4*sm4;
            const long double sm9 = sm4*sm5;
            const long double sm10 = sm5*sm5;

            const long double sp2 = sp*sp;
            const long double sp3 = sp*sp2;
            const long double sp4 = sp2*sp2;
            const long double sp5 = sp2*sp3;
            const long double sp6 = sp3*sp3;
            const long double sp7 = sp3*sp4;
            const long double sp8 = sp4*sp4;
            const long double sp9 = sp4*sp5;
            const long double sp10 = sp5*sp5;

            const long double denom = 2*sm2*sp2*ssum4*(5*sm4 - 10*sm3*sp + 12*sm2*sp2 - 10*sm*sp3 + 5*sp4);
            assert(denom > 0.0);

            a_ = (3*sdiff2*(5*sm6 + 8*sm5*sp + 5*sm4*sp2 + 8*sm3*sp3 + 5*sm2*sp4 + 8*sm*sp5 + 5*sp6))/denom;
            b_ = (sdiff*(25*sm8 + 14*sm7*sp - 14*sm6*sp2 + 14*sm5*sp3 - 14*sm4*sp4 + 14*sm3*sp5 - 14*sm2*sp6 + 14*sm*sp7 + 25*sp8))/denom;
            c_ = (10*sm10 - 5*sm9*sp + 30*sm7*sp3 - 6*sm6*sp4 + 6*sm5*sp5 - 6*sm4*sp6 + 30*sm3*sp7 - 5*sm*sp9 + 10*sp10)/denom;

            // Make sure that we have a maximum at 0
            assert(c_ > 0.0);

            // Make sure that the curve has a single extremum
            const double det = 9*b_*b_ - 32*a_*c_;
            assert(det < 0.0);
        }
    }

    double MoldedQuartic::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double MoldedQuartic::uValue(const double x) const
    {
        return -0.5*x*x*((a_*x + b_)*x + c_);
    }

    double MoldedQuartic::uDerivative(const double x) const
    {
        return -0.5*x*((4.0*a_*x + 3.0*b_)*x + 2.0*c_);
    }

    double MoldedQuartic::uSecondDerivative(const double x, double) const
    {
        return -0.5*((12.0*a_*x + 6.0*b_)*x + 2.0*c_);
    }

    double MoldedQuartic::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double MoldedQuartic::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    MatchedQuintic::MatchedQuintic(const double i_location,
                                   const double i_sigmaPlus,
                                   const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          a_(0.0), b_(0.0), c_(0.0), d_(0.0)
    {
        static const double maxRatio = 2.42641998607398;

        validateSigmas("ase::MatchedQuintic constructor",
                       sigPlus_, sigMinus_);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio > maxRatio || 1.0/sigratio > maxRatio)
            throw std::invalid_argument(
                "In ase::MatchedQuintic constructor: "
                "sigma asymmetry is too high");

        if (sigMinus_ == sigPlus_)
            d_ = 1.0/sigPlus_/sigPlus_;
        else
        {
            const long double sm = sigMinus_;
            const long double sp = sigPlus_;

            const long double sprod = sm*sp;
            const long double sprod2 = sprod*sprod;

            const long double sdiff = sm - sp;
            const long double sdiff2 = sdiff*sdiff;

            const long double sm2 = sm*sm;
            const long double sm3 = sm*sm2;
            const long double sm4 = sm2*sm2;

            const long double sp2 = sp*sp;
            const long double sp3 = sp*sp2;
            const long double sp4 = sp2*sp2;

            const long double denom = sprod2*(8*sm2 + 19*sprod + 8*sp2);

            a_ = -10*sdiff/denom;
            b_ = -18*sdiff2/denom;
            c_ = 45*sprod*sdiff/denom;
            d_ = (8*sm4 + 19*sm3*sp - 19*sm2*sp2 + 19*sm*sp3 + 8*sp4)/denom;

            // Make sure that we have a maximum at 0
            assert(uSecondDerivative(0.0, 0.0) < 0.0);

            // Make sure that the curve has a single extremum
            double coeffs[6] = {0.0};
            coeffs[2] = d_;
            coeffs[3] = c_;
            coeffs[4] = b_;
            coeffs[5] = a_;
            const Poly1D poly(coeffs, 5);
            const Poly1D& deriv = poly.derivative();
            assert(deriv.nRoots(-sigMinus_, sigPlus_) == 1U);
        }

        dplus_ = quinticDeriv(sigPlus_);
        dminus_ = quinticDeriv(-sigMinus_);
    }

    double MatchedQuintic::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double MatchedQuintic::quinticValue(const double x) const
    {
        const double x2 = x*x;
        return -0.5*x2*((a_*x + b_)*x2 + c_*x + d_);
    }

    double MatchedQuintic::quinticDeriv(const double x) const
    {
        return -0.5*x*((5*a_*x + 4*b_)*x*x + 3*c_*x + 2*d_);
    }

    double MatchedQuintic::quinticSecondDeriv(const double x) const
    {
        return -0.5*((20*a_*x + 12*b_)*x*x + 6*c_*x + 2*d_);
    }

    double MatchedQuintic::uValue(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return del*(der2*del/2.0 + der) - 0.5;
        }
        else
            return quinticValue(x);
    }

    double MatchedQuintic::uDerivative(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return der2*del + der;
        }
        else
            return quinticDeriv(x);
    }

    double MatchedQuintic::uSecondDerivative(const double x, double) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            return -1.0/sig/sig;
        }
        else
            return quinticSecondDeriv(x);
    }

    double MatchedQuintic::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double MatchedQuintic::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    DoubleQuintic::DoubleQuintic(const double i_location,
                                 const double i_sigmaPlus,
                                 const double i_sigmaMinus,
                                 const double s02)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          aleft_(0.0), bleft_(0.0), cleft_(0.0),
          aright_(0.0), bright_(0.0), cright_(0.0), d_(0.0)
    {
        validateSigmas("ase::DoubleQuintic constructor",
                       sigPlus_, sigMinus_);
        if (s02 <= 0.0) throw std::invalid_argument(
            "In ase::DoubleQuintic constructor: "
            "effective variance parameter must be positive");

        d_ = 1.0/s02;

        const double sp2 = sigPlus_*sigPlus_;
        const double sp3 = sigPlus_*sp2;
        const double sp5 = sp2*sp3;
        const double sm2 = sigMinus_*sigMinus_;
        const double sm3 = sigMinus_*sm2;
        const double sm5 = sm2*sm3;

        {
            const double sigd = sp2 - s02;
            aright_ = -sigd/s02/sp5;
            cright_ = -3.0*sigd/s02/sp3;
            bright_ = -cright_/sigPlus_;
        }

        {
            const double sigd = sm2 - s02;
            aleft_ = sigd/s02/sm5;
            cleft_ = 3.0*sigd/s02/sm3;
            bleft_ = cleft_/sigMinus_;
        }

        // Make sure that the curve has a single extremum
        const double sigFrac = 1.0e-5;
        double coeffs[6] = {0.0};
        coeffs[2] = d_;

        coeffs[3] = cleft_;
        coeffs[4] = bleft_;
        coeffs[5] = aleft_;
        {
            const Poly1D poly(coeffs, 5);
            const Poly1D& deriv = poly.derivative();
            if (deriv.nRoots(-sigMinus_, -sigMinus_*sigFrac))
                throw std::invalid_argument(
                    "In ase::DoubleQuintic constructor: "
                    "the curve has an extra extremum on [-sigmaMinus, 0]");
        }

        coeffs[3] = cright_;
        coeffs[4] = bright_;
        coeffs[5] = aright_;
        {
            const Poly1D poly(coeffs, 5);
            const Poly1D& deriv = poly.derivative();
            if (deriv.nRoots(sigPlus_*sigFrac, sigPlus_))
                throw std::invalid_argument(
                    "In ase::DoubleQuintic constructor: "
                    "the curve has an extra extremum on [0, sigmaPlus]");
        }

        dplus_ = quinticDeriv(sigPlus_);
        dminus_ = quinticDeriv(-sigMinus_);
    }

    double DoubleQuintic::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double DoubleQuintic::quinticValue(const double x) const
    {
        double a = aright_, b = bright_, c = cright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
            c = cleft_;
        }
        const double x2 = x*x;
        return -0.5*x2*((a*x + b)*x2 + c*x + d_);
    }

    double DoubleQuintic::quinticDeriv(const double x) const
    {
        double a = aright_, b = bright_, c = cright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
            c = cleft_;
        }
        return -0.5*x*((5*a*x + 4*b)*x*x + 3*c*x + 2*d_);
    }

    double DoubleQuintic::quinticSecondDeriv(const double x) const
    {
        double a = aright_, b = bright_, c = cright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
            c = cleft_;
        }
        return -0.5*((20*a*x + 12*b)*x*x + 6*c*x + 2*d_);
    }

    double DoubleQuintic::uValue(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return del*(der2*del/2.0 + der) - 0.5;
        }
        else
            return quinticValue(x);
    }

    double DoubleQuintic::uDerivative(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return der2*del + der;
        }
        else
            return quinticDeriv(x);
    }

    double DoubleQuintic::uSecondDerivative(const double x, double) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            return -1.0/sig/sig;
        }
        else
            return quinticSecondDeriv(x);
    }

    double DoubleQuintic::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double DoubleQuintic::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    DoubleQuartic::DoubleQuartic(const double i_location,
                                 const double i_sigmaPlus,
                                 const double i_sigmaMinus,
                                 const double s02)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          aleft_(0.0), bleft_(0.0),
          aright_(0.0), bright_(0.0), c_(0.0)
    {
        validateSigmas("ase::DoubleQuartic constructor",
                       sigPlus_, sigMinus_);
        if (s02 <= 0.0) throw std::invalid_argument(
            "In ase::DoubleQuartic constructor: "
            "effective variance parameter must be positive");

        c_ = 1.0/s02;

        const double sp2 = sigPlus_*sigPlus_;
        const double sp3 = sigPlus_*sp2;
        const double sp4 = sp2*sp2;
        const double sm2 = sigMinus_*sigMinus_;
        const double sm3 = sigMinus_*sm2;
        const double sm4 = sm2*sm2;

        {
            const double sigd = sp2 - s02;
            aright_ = 2.0/3.0*sigd/s02/sp4;
            bright_ = -5.0/3.0*sigd/s02/sp3;
        }

        {
            const double sigd = sm2 - s02;
            aleft_ = 2.0/3.0*sigd/s02/sm4;
            bleft_ = 5.0/3.0*sigd/s02/sm3;
        }

        // Make sure that the curve has a single extremum
        const double sigFrac = 1.0e-5;
        double coeffs[5] = {0.0};
        coeffs[2] = c_;

        coeffs[3] = bleft_;
        coeffs[4] = aleft_;
        {
            const Poly1D poly(coeffs, 4);
            const Poly1D& deriv = poly.derivative();
            if (deriv.nRoots(-sigMinus_, -sigMinus_*sigFrac))
                throw std::invalid_argument(
                    "In ase::DoubleQuartic constructor: "
                    "the curve has an extra extremum on [-sigmaMinus, 0]");
        }

        coeffs[3] = bright_;
        coeffs[4] = aright_;
        {
            const Poly1D poly(coeffs, 4);
            const Poly1D& deriv = poly.derivative();
            if (deriv.nRoots(sigPlus_*sigFrac, sigPlus_))
                throw std::invalid_argument(
                    "In ase::DoubleQuartic constructor: "
                    "the curve has an extra extremum on [0, sigmaPlus]");
        }

        dplus_ = quarticDeriv(sigPlus_);
        dminus_ = quarticDeriv(-sigMinus_);
    }

    double DoubleQuartic::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double DoubleQuartic::quarticValue(const double x) const
    {
        double a = aright_, b = bright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
        }
        return -0.5*x*x*(c_ + x*(b + a*x));
    }

    double DoubleQuartic::quarticDeriv(const double x) const
    {
        double a = aright_, b = bright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
        }
        return -0.5*x*(2.0*c_ + x*(3.0*b + 4.0*a*x));
    }

    double DoubleQuartic::quarticSecondDeriv(const double x) const
    {
        double a = aright_, b = bright_;
        if (x < 0.0)
        {
            a = aleft_;
            b = bleft_;
        }
        return -(c_ + 3.0*x*(b + 2.0*a*x));
    }

    double DoubleQuartic::uValue(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return del*(der2*del/2.0 + der) - 0.5;
        }
        else
            return quarticValue(x);
    }

    double DoubleQuartic::uDerivative(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
           const double der = x >= 0.0 ? dplus_ : dminus_;
           const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
           const double del = x >= 0.0 ? x - sigPlus_ : x + sigMinus_;
           const double der2 = -1.0/sig/sig;
           return der2*del + der;
        }
        else
            return quarticDeriv(x);
    }

    double DoubleQuartic::uSecondDerivative(const double x, double) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            return -1.0/sig/sig;
        }
        else
            return quarticSecondDeriv(x);
    }

    double DoubleQuartic::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double DoubleQuartic::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    Interpolated7thDegree::Interpolated7thDegree(const double i_location,
                                           const double i_sigmaPlus,
                                           const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        static const double maxRatio = 2.744405155225988;

        validateSigmas("ase::Interpolated7thDegree constructor",
                       sigPlus_, sigMinus_);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio > maxRatio || 1.0/sigratio > maxRatio)
            throw std::invalid_argument(
                "In ase::Interpolated7thDegree constructor: "
                "sigma asymmetry is too high");

        long double coeffs[8] = {0.0L};

        if (sigMinus_ == sigPlus_)
        {
            coeffs[2] = 1.0/sigPlus_/sigPlus_;
            septic_ = Poly1D(coeffs, 2);
        }
        else
        {
            const long double sm = sigMinus_;
            const long double sp = sigPlus_;

            const long double ssum = sm + sp;
            const long double ssum2 = ssum*ssum;
            const long double ssum4 = ssum2*ssum2;

            const long double sprod = sm*sp;
            const long double sprod2 = sprod*sprod;
            
            const long double sdiff = sm - sp;
            const long double sdiff2 = sdiff*sdiff;

            const long double sm2 = sm*sm;
            const long double sm3 = sm*sm2;
            const long double sm4 = sm2*sm2;
            const long double sm5 = sm2*sm3;
            const long double sm6 = sm3*sm3;

            const long double sp2 = sp*sp;
            const long double sp3 = sp*sp2;
            const long double sp4 = sp2*sp2;
            const long double sp5 = sp2*sp3;
            const long double sp6 = sp3*sp3;

            const long double denom = sprod2*ssum4;
            coeffs[2] = (sm6 + 4*sm5*sp + 6*sm4*sp2 - 6*sm3*sp3 + 6*sm2*sp4 + 4*sm*sp5 + sp6)/denom;
            coeffs[3] = 30*sprod2*sdiff/denom;
            coeffs[4] = -30*sprod*sdiff2/denom;
            coeffs[5] = 10*sdiff*(sm2 - 4*sprod + sp2)/denom;
            coeffs[6] = 15*sdiff2/denom;
            coeffs[7] = 6*sdiff/denom;
            septic_ = Poly1D(coeffs, 7);
        }
        deriv_ = septic_.derivative();
        secondDeriv_ = deriv_.derivative();

        // Make sure we have a maximum at 0
        assert(uSecondDerivative(0.0, 0.0) < 0.0);

        // Make sure we don't have other extrema
        assert(deriv_.nRoots(-sigMinus_, sigPlus_) == 1U);
    }

    double Interpolated7thDegree::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double Interpolated7thDegree::uValue(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            const double r = x/sig;
            return -0.5*r*r;
        }
        else
            return -0.5*septic_(x);
    }

    double Interpolated7thDegree::uDerivative(const double x) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            return -x/sig/sig;
        }
        else
            return -0.5*deriv_(x);
    }

    double Interpolated7thDegree::uSecondDerivative(const double x, double) const
    {
        if (x <= -sigMinus_ || x >= sigPlus_)
        {
            const double sig = x >= 0.0 ? sigPlus_ : sigMinus_;
            return -1.0/sig/sig;
        }
        else
            return -0.5*secondDeriv_(x);
    }

    double Interpolated7thDegree::uSigmaPlus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double Interpolated7thDegree::uSigmaMinus(const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    VariableSigmaLogli::VariableSigmaLogli(const double i_location,
                                           const double i_sigmaPlus,
                                           const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::VariableSigmaLogli constructor",
                       sigPlus_, sigMinus_);
        const double sigsum = sigPlus_ + sigMinus_;
        sigma0_ = 2.0*sigPlus_*sigMinus_/sigsum;
        sigmaPrime_ = (sigPlus_ - sigMinus_)/sigsum;
        if (sigmaPrime_ == 0.0)
        {
            pmin_ = -DBL_MAX;
            pmax_ = DBL_MAX;
        }
        else
        {
            const double lim = -(1.0 - DBL_EPSILON)*sigma0_/sigmaPrime_;
            if (sigmaPrime_ > 0.0)
            {
                pmin_ = lim;
                pmax_ = DBL_MAX;
            }
            else
            {
                pmin_ = -DBL_MAX;
                pmax_ = lim;
            }
            truncationLogli_ = uValue(lim);
        }
    }

    double VariableSigmaLogli::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double VariableSigmaLogli::uValue(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableSigmaLogli::uValue: "
            "argument is out of range");
        const double del = x/(sigma0_ + sigmaPrime_*x);
        return -del*del/2.0;
    }

    double VariableSigmaLogli::uDerivative(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableSigmaLogli::uDerivative: "
            "argument is out of range");
        const double tmp = sigma0_ + sigmaPrime_*x;
        return -sigma0_*x/tmp/tmp/tmp;
    }

    double VariableSigmaLogli::uSecondDerivative(const double x, double) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableSigmaLogli::uSecondDerivative: "
            "argument is out of range");
        const double tmp = sigma0_ + sigmaPrime_*x;
        return -sigma0_*(sigma0_ - 2.0*sigmaPrime_*x)/tmp/tmp/tmp/tmp;
    }

    double VariableSigmaLogli::uSigmaPlus(const double deltaLogli,
                                          double) const
    {
        if (sigmaPrime_ < 0.0 && deltaLogli >= -truncationLogli_)
            throw std::invalid_argument(
                "In ase::VariableSigmaLogli::uSigmaPlus: "
                "deltaLogLikelihood argument is too large");
        const double tmp = sqrt(2.0*deltaLogli);
        return tmp*sigma0_/(1.0 - tmp*sigmaPrime_);
    }

    double VariableSigmaLogli::uSigmaMinus(const double deltaLogli,
                                           double) const
    {
        if (sigmaPrime_ > 0.0 && deltaLogli >= -truncationLogli_)
            throw std::invalid_argument(
                "In ase::VariableSigmaLogli::uSigmaMinus: "
                "deltaLogLikelihood argument is too large");
        const double tmp = -sqrt(2.0*deltaLogli);
        return tmp*sigma0_/(tmp*sigmaPrime_ - 1.0);
    }

    /*********************************************************************/

    LogLogisticBeta::LogLogisticBeta(const double i_location,
                                     const double i_sigmaPlus,
                                     const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          a_(0.0), c_(0.0), g_(0.0),
          symp_(0.0, i_sigmaPlus, i_sigmaPlus)
    {
        validateSigmas("ase::LogLogisticBeta constructor",
                       sigPlus_, sigMinus_);
        if (sigPlus_ != sigMinus_)
        {
            const double sp = std::max(sigPlus_, sigMinus_);
            const double sm = std::min(sigPlus_, sigMinus_);
            const std::pair<double,double>& opt = llc_optimizeLogisticBeta(sp, sm);
            const double v = llc_logLogisticBeta(sp, opt.first, opt.second, 1.0);
            a_ = sigPlus_ > sigMinus_ ? opt.first : -opt.first;
            c_ = opt.second;
            g_ = -0.5/v;
        }
    }

    double LogLogisticBeta::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double LogLogisticBeta::uValue(const double x) const
    {
        if (sigPlus_ == sigMinus_)
            return symp_(x);
        else
            return llc_logLogisticBeta(x, a_, c_, g_);
    }

    double LogLogisticBeta::uDerivative(const double x) const
    {
        if (sigPlus_ == sigMinus_)
            return symp_.derivative(x);
        else
        {
            const double tmp = exp(x/c_);
            return ((a_*a_ - 1)*c_*(tmp - 1)*g_)/(1 + tmp + a_*(tmp - 1));
        }
    }

    double LogLogisticBeta::uSecondDerivative(const double x, double /* step */) const
    {
        if (sigPlus_ == sigMinus_)
            return symp_.secondDerivative(x);
        else
        {
            const double tmp = exp(x/c_);
            const double denom = 1 + tmp + a_*(tmp - 1);
            return 2*(a_*a_ - 1)*tmp*g_/denom/denom;
        }
    }

    double LogLogisticBeta::uSigmaPlus(
        const double deltaLogli, const double f) const
    {
        if (sigPlus_ == sigMinus_)
            return symp_.sigmaPlus(deltaLogli, f);
        else
            return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double LogLogisticBeta::uSigmaMinus(
        const double deltaLogli, const double f) const
    {
        if (sigPlus_ == sigMinus_)
            return symp_.sigmaMinus(deltaLogli, f);
        else
            return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    PDGLogli::PDGLogli(const double i_location,
                       const double i_sigmaPlus,
                       const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::PDGLogli constructor",
                       sigPlus_, sigMinus_);
        const double sigsum = sigPlus_ + sigMinus_;
        sigma0_ = 2.0*sigPlus_*sigMinus_/sigsum;
        sigmaPrime_ = (sigPlus_ - sigMinus_)/sigsum;
    }

    double PDGLogli::sigmaValue(const double x) const
    {
        if (x <= -sigMinus_)
            return sigMinus_;
        else if (x >= sigPlus_)
            return sigPlus_;
        else
            return sigma0_ + sigmaPrime_*x;
    }

    double PDGLogli::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double PDGLogli::uValue(const double x) const
    {
        const double del = x/sigmaValue(x);
        return -del*del/2.0;
    }

    double PDGLogli::uDerivative(const double x) const
    {
        const double sig = sigmaValue(x);
        if (x <= -sigMinus_ || x >= sigPlus_)
            return -x/sig/sig;
        else
            return -sigma0_*x/sig/sig/sig;
    }

    double PDGLogli::uSecondDerivative(const double x, double) const
    {
        const double tmp = sigmaValue(x);
        if (x <= -sigMinus_ || x >= sigPlus_)
            return -1.0/tmp/tmp;
        else
            return -sigma0_*(sigma0_ - 2.0*sigmaPrime_*x)/tmp/tmp/tmp/tmp;
    }

    double PDGLogli::uSigmaPlus(const double deltaLogli,
                                double) const
    {
        const double tmp = sqrt(2.0*deltaLogli);
        if (tmp >= 1.0)
            return tmp*sigPlus_;
        else
            return tmp*sigma0_/(1.0 - tmp*sigmaPrime_);
    }

    double PDGLogli::uSigmaMinus(const double deltaLogli,
                                 double) const
    {
        const double tmp = -sqrt(2.0*deltaLogli);
        if (tmp <= -1.0)
            return -tmp*sigMinus_;
        else
            return tmp*sigma0_/(tmp*sigmaPrime_ - 1.0);
    }

    /*********************************************************************/

    VariableVarianceLogli::VariableVarianceLogli(const double i_location,
                                                 const double i_sigmaPlus,
                                                 const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        validateSigmas("ase::VariableVarianceLogli constructor",
                       sigPlus_, sigMinus_);
        v0_ = sigPlus_*sigMinus_;
        vPrime_ = sigPlus_ - sigMinus_;
        if (vPrime_ == 0.0)
        {
            pmin_ = -DBL_MAX;
            pmax_ = DBL_MAX;
        }
        else
        {
            const double lim = -(1.0 - DBL_EPSILON)*v0_/vPrime_;
            if (vPrime_ > 0.0)
            {
                pmin_ = lim;
                pmax_ = DBL_MAX;
            }
            else
            {
                pmin_ = -DBL_MAX;
                pmax_ = lim;
            }
            truncationLogli_ = uValue(lim);
        }
    }

    double VariableVarianceLogli::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double VariableVarianceLogli::uValue(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableVarianceLogli::uValue: "
            "argument is out of range");
        const double v = v0_ + vPrime_*x;
        return -x*x/v/2.0;
    }

    double VariableVarianceLogli::uDerivative(const double x) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableVarianceLogli::uDerivative: "
            "argument is out of range");
        const double v = v0_ + vPrime_*x;
        return -x*(2.0*v0_ + vPrime_*x)/v/v/2.0;
    }

    double VariableVarianceLogli::uSecondDerivative(const double x, double) const
    {
        if (x < pmin_ || x > pmax_) throw std::invalid_argument(
            "In ase::VariableVarianceLogli::uSecondDerivative: "
            "argument is out of range");
        const double v = v0_ + vPrime_*x;
        return -v0_*v0_/v/v/v;
    }

    double VariableVarianceLogli::uSigmaPlus(const double deltaLogli,
                                             double) const
    {
        if (vPrime_ < 0.0 && deltaLogli >= -truncationLogli_)
            throw std::invalid_argument(
                "In ase::VariableVarianceLogli::uSigmaPlus: "
                "deltaLogLikelihood argument is too large");
        if (vPrime_ == 0.0)
            return sqrt(2.0*deltaLogli*v0_);
        else
        {
            double r1, r2;
            const unsigned nSols = solveQuadratic(
                -2.0*deltaLogli*vPrime_, -2.0*deltaLogli*v0_, &r1, &r2);
            assert(nSols == 2U);
            assert(r1*r2 < 0.0);
            if (r1 > 0.0)
                return r1;
            else
                return r2;
        }
    }

    double VariableVarianceLogli::uSigmaMinus(const double deltaLogli,
                                              double) const
    {
        if (vPrime_ > 0.0 && deltaLogli >= -truncationLogli_)
            throw std::invalid_argument(
                "In ase::VariableVarianceLogli::uSigmaMinus: "
                "deltaLogLikelihood argument is too large");
        if (vPrime_ == 0.0)
            return sqrt(2.0*deltaLogli*v0_);
        else
        {
            double r1, r2;
            const unsigned nSols = solveQuadratic(
                -2.0*deltaLogli*vPrime_, -2.0*deltaLogli*v0_, &r1, &r2);
            assert(nSols == 2U);
            assert(r1*r2 < 0.0);
            if (r1 > 0.0)
                return -r2;
            else
                return -r1;
        }
    }

    /*********************************************************************/

    VariableLogSigma::VariableLogSigma(const double i_location,
                                       const double i_sigmaPlus,
                                       const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus)
    {
        static const double maxRatio = 5.33845389008676;
        
        validateSigmas("ase::VariableLogSigma constructor",
                       sigPlus_, sigMinus_);
        const double sigratio = sigPlus_/sigMinus_;
        if (sigratio > maxRatio || 1.0/sigratio > maxRatio)
            throw std::invalid_argument(
                "In ase::VariableLogSigma constructor: "
                "sigma asymmetry is too high");

        logPlus_ = log(sigPlus_);
        logMinus_ = log(sigMinus_);
    }

    double VariableLogSigma::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double VariableLogSigma::sigmaValue(const double x) const
    {
        if (sigPlus_ == sigMinus_)
            return sigPlus_;
        else
        {
            const FechnerDistribution fech(0.0, sigPlus_, sigMinus_);
            const double cdfLeft = fech.cdf(-sigMinus_);
            const double cdfRight = fech.cdf(sigPlus_);
            const double cdf = fech.cdf(x);
            const double logSig = linearValue(cdfLeft, logMinus_, cdfRight,
                                              logPlus_, cdf);
            return exp(logSig);
        }
    }

    void VariableLogSigma::sigmaDerivative(
        const double x, double* value, double* deriv) const
    {
        assert(value);
        assert(deriv);

        if (sigPlus_ == sigMinus_)
        {
            *value = sigPlus_;
            *deriv = 0.0;
        }
        else
        {
            const FechnerDistribution fech(0.0, sigPlus_, sigMinus_);
            const double cdfLeft = fech.cdf(-sigMinus_);
            const double cdfRight = fech.cdf(sigPlus_);
            const double cdf = fech.cdf(x);
            const double logSig = linearValue(cdfLeft, logMinus_, cdfRight,
                                              logPlus_, cdf);
            *value = exp(logSig);
            const double slope = (logPlus_ - logMinus_)/(cdfRight - cdfLeft);
            *deriv = *value * slope * fech.density(x);
        }
    }

    void VariableLogSigma::sigmaSecondDerivative(
        const double x, double* value,
        double* deriv, double* secondDeriv) const
    {
        assert(value);
        assert(deriv);
        assert(secondDeriv);

        if (sigPlus_ == sigMinus_)
        {
            *value = sigPlus_;
            *deriv = 0.0;
            *secondDeriv = 0.0;
        }
        else
        {
            const FechnerDistribution fech(0.0, sigPlus_, sigMinus_);
            const double cdfLeft = fech.cdf(-sigMinus_);
            const double cdfRight = fech.cdf(sigPlus_);
            const double cdf = fech.cdf(x);
            const double logSig = linearValue(cdfLeft, logMinus_, cdfRight,
                                              logPlus_, cdf);
            *value = exp(logSig);
            const double slope = (logPlus_ - logMinus_)/(cdfRight - cdfLeft);
            const double dens = fech.density(x);
            *deriv = *value * slope * dens;
            *secondDeriv = slope*(*deriv*dens + *value*fech.densityDerivative(x));
        }
    }

    double VariableLogSigma::uValue(const double x) const
    {
        const double del = x/sigmaValue(x);
        return -del*del/2.0;
    }

    double VariableLogSigma::uDerivative(const double x) const
    {
        double sig, deriv;
        sigmaDerivative(x, &sig, &deriv);
        const double del = x/sig;
        return -del*(sig - x*deriv)/sig/sig;
    }

/*
    bool VariableLogSigma::hasSecondExtremum() const
    {
        const double sp = std::max(sigPlus_, sigMinus_);
        const double sm = std::min(sigPlus_, sigMinus_);
        const double logp = log(sp);
        const double logm = log(sm);
        const FechnerDistribution fech(0.0, sp, sm);
        const double cdfLeft = fech.cdf(-sm);
        const double cdfRight = fech.cdf(sp);
        const double slope = (logp - logm)/(cdfRight - cdfLeft);
        return sp*slope*fech.density(sp) >= 1.0;
    }
*/

    double VariableLogSigma::uSecondDerivative(const double x, double) const
    {
        double sig, deriv, sder;
        sigmaSecondDerivative(x, &sig, &deriv, &sder);
        const double sig2 = sig*sig;
        const double sig4 = sig2*sig2;
        return (x*sig*(4*deriv + x*sder) - sig2 - 3*x*x*deriv*deriv)/sig4;
    }

    double VariableLogSigma::uSigmaPlus(const double deltaLogli,
                                        const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double VariableLogSigma::uSigmaMinus(const double deltaLogli,
                                         const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    DoubleCubicLogSigma::DoubleCubicLogSigma(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus,
                                             const double sig0)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          interp_(1.0/sig0/sig0, i_sigmaPlus, 1.0/i_sigmaPlus/i_sigmaPlus,
                  i_sigmaMinus, 1.0/i_sigmaMinus/i_sigmaMinus)
    {
    }

    double DoubleCubicLogSigma::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double DoubleCubicLogSigma::uValue(const double x) const
    {
        return -x*x/2.0*interp_(x);
    }

    double DoubleCubicLogSigma::uDerivative(const double x) const
    {
        return -x*(interp_(x) + x/2.0*interp_.derivative(x));
    }

    double DoubleCubicLogSigma::uSecondDerivative(const double x, double) const
    {
        return -(interp_(x) + x*(2.0*interp_.derivative(x) +
                                 x/2.0*interp_.secondDerivative(x)));
    }

    double DoubleCubicLogSigma::uSigmaPlus(const double deltaLogli,
                                           const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double DoubleCubicLogSigma::uSigmaMinus(const double deltaLogli,
                                            const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    QuinticLogSigma::QuinticLogSigma(const double i_location,
                                     const double i_sigmaPlus,
                                     const double i_sigmaMinus)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          interp_(i_sigmaPlus, 1.0/i_sigmaPlus/i_sigmaPlus,
                  i_sigmaMinus, 1.0/i_sigmaMinus/i_sigmaMinus)
    {
        if (hasExtraExtremum()) throw std::invalid_argument(
            "In ase::QuinticLogSigma constructor: "
            "sigma asymmetry is too high");
    }

    bool QuinticLogSigma::hasExtraExtremum() const
    {
        const Poly1D& inDer = interp_.innerDerivPoly();
        double coeffs[6];
        for (unsigned deg=0; deg<5U; ++deg)
            coeffs[deg+1U] = inDer[deg]/2.0;
        coeffs[0] = 1.0;
        const Poly1D detPoly(coeffs, 5);
        return detPoly.nRoots(-sigMinus_, sigPlus_);
    }

    double QuinticLogSigma::stepSize() const
    {
        return 0.01*std::min(sigPlus_, sigMinus_);
    }

    double QuinticLogSigma::uValue(const double x) const
    {
        return -x*x/2.0*interp_(x);
    }

    double QuinticLogSigma::uDerivative(const double x) const
    {
        return -x*(interp_(x) + x/2.0*interp_.derivative(x));
    }

    double QuinticLogSigma::uSecondDerivative(const double x, double) const
    {
        return -(interp_(x) + x*(2.0*interp_.derivative(x) +
                                 x/2.0*interp_.secondDerivative(x)));
    }

    double QuinticLogSigma::uSigmaPlus(const double deltaLogli,
                                           const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double QuinticLogSigma::uSigmaMinus(const double deltaLogli,
                                            const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    ConservativeSpline::ConservativeSpline(const double i_location,
                                           const double i_sigmaPlus,
                                           const double i_sigmaMinus,
                                           const double secondDerivFactor)
        : AbsShiftableLogli(i_location),
          sigPlus_(i_sigmaPlus), sigMinus_(i_sigmaMinus),
          ksq_(secondDerivFactor),
          s0sq_(i_sigmaPlus*i_sigmaMinus),
          a_(0.0),
          c_(2.0/i_sigmaPlus),
          d_(-2.0/i_sigmaMinus),
          rx0_(0.0),
          lx0_(0.0)
    {
        validateSigmas("ase::ConservativeSpline constructor",
                       sigPlus_, sigMinus_);
        if (ksq_ < 1.0)
            throw std::invalid_argument(
                "In ase::ConservativeSpline constructor: "
                "factor for second derivative limit can not be less than 1");
        else if (ksq_ > 1.0)
        {
            const double ksqLimit = maxDerivLimitFactor(sigPlus_, sigMinus_);
            if (ksq_ > ksqLimit)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::ConservativeSpline constructor: "
                   << "factor for the second derivative limit is too large. "
                   << "For these uncertainties it must be at most "
                   << ksqLimit << '.';
                throw std::invalid_argument(os.str());
            }
            const bool posRootSp = sigPlus_ < sigMinus_;
            const bool posRootSm = posRootSp;
            const std::pair<double,double>& lims = s0sqRange(sigPlus_, sigMinus_, ksq_);
            if (lims.first == lims.second)
                s0sq_ = lims.first;
            else
            {
                const ConservativeSplineADelta adel(sigPlus_, sigMinus_, ksq_,
                                                    posRootSp, posRootSm);
                const double lo = lims.first*(1.0 + tol_);
                const double up = lims.second*(1.0 - tol_);
                const double status = findRootUsingBisections(adel, 0.0, lo, up, tol_, &s0sq_);
                if (!status) throw std::runtime_error(
                    "In ase::ConservativeSpline constructor: root finding failed");
            }
            const double ksqplus = sigPlus_ > sigMinus_ ? 1.0/ksq_ : ksq_;
            const double ksqminus = sigPlus_ > sigMinus_ ? ksq_ : 1.0/ksq_;
            a_ = aformula(ksqplus, s0sq_, sigPlus_, posRootSp);
            c_ = cformula(ksqplus, s0sq_, sigPlus_, posRootSp);
            d_ = cformula(ksqminus, s0sq_, -sigMinus_, posRootSm);
            rx0_ = rxformula(ksqplus, s0sq_, sigPlus_, posRootSp);
            lx0_ = rxformula(ksqminus, s0sq_, -sigMinus_, posRootSm);
        }
    }

    double ConservativeSpline::maxDerivLimitFactor(const double sp, const double sm)
    {
        validateSigmas("ase::ConservativeSpline::maxDerivLimitFactor", sp, sm);
        if (sp == sm)
            return 1.0;
        const double kmax = (1.0 - 100.0*tol_)*sqrtDerivLimitFactor(sp, sm);
        if (kmax <= 1.0)
            return 1.0;
        if (isKsqUsable(sp, sm, kmax))
            return kmax;
        double kUsable = 1.0;
        double kUnUsable = kmax;
        for (unsigned i=0; i<1000; ++i)
        {
            const double ktry = (kUsable + kUnUsable)/2.0;
            if (isKsqUsable(sp, sm, ktry))
                kUsable = ktry;
            else
                kUnUsable = ktry;
            if ((kUnUsable - kUsable)/kUsable < tol_)
                return kUsable;
        }
        throw std::runtime_error("In ase::ConservativeSpline::maxDerivLimitFactor: "
                                 "limit derivation failed");
        return 0.0;
    }

    bool ConservativeSpline::isKsqUsable(const double sp, const double sm,
                                         const double ksq)
    {
        const bool posRootSp = sp < sm;
        const bool posRootSm = posRootSp;
        const std::pair<double,double>& lims = s0sqRange(sp, sm, ksq);
        const double lo = lims.first*(1.0 + tol_);
        const double up = lims.second*(1.0 - tol_);
        const ConservativeSplineADelta adel(sp, sm, ksq, posRootSp, posRootSm);
        if (adel(lo)*adel(up) <= 0.0)
        {
            double s0sq;
            const double status = findRootUsingBisections(adel, 0.0, lo, up, tol_, &s0sq);
            assert(status);
            const double ksqplus = sp > sm ? 1.0/ksq : ksq;
            const double ksqminus = sp > sm ? ksq : 1.0/ksq;
            const double rx0 = rxformula(ksqplus, s0sq, sp, posRootSp);
            const double lx0 = rxformula(ksqminus, s0sq, -sm, posRootSm);
            return rx0 <= sp && std::abs(lx0) <= sm;
        }
        else
            return false;
    }

    double ConservativeSpline::adelta(const double ksq, const double s0sq,
                                      const double sp, const double sm,
                                      const bool posRootForSp, const bool posRootForSm)
    {
        if (ksq < 1.0) throw std::invalid_argument(
            "In ase::ConservativeSpline::adelta: "
            "ksq argument can not be less than 1");
        const std::pair<double,double>& lims = s0sqRange(sp, sm, ksq);
        assert(lims.first <= lims.second);
        if (s0sq < lims.first || s0sq > lims.second)
        {
            std::ostringstream os;
            os.precision(16);
            os << "In ase::ConservativeSpline::adelta: "
               << "s0sq argument " << s0sq << " is outside of allowed range "
               << '[' << lims.first << ", " << lims.second << ']';
            throw std::runtime_error(os.str());
        }
        double ksqplus, ksqminus;
        if (sp > sm)
        {
            ksqplus = 1.0/ksq;
            ksqminus = ksq;
        }
        else
        {
            ksqplus = ksq;
            ksqminus = 1.0/ksq;
        }
        return aformula(ksqplus, s0sq, sp, posRootForSp) -
               aformula(ksqminus, s0sq, -sm, posRootForSm);
    }

    double ConservativeSpline::sqrformula(const double ksq, const double s0sq,
                                          const double sp)
    {
        // Checked with Mathematica
        const double sp2 = sp*sp;
        const double s0sq2 = s0sq*s0sq;
        const double tmp = ksq*(4 - ksq)*s0sq2  + sp2*(3*sp2 - 2*(2 + ksq)*s0sq);
        if (tmp < 0.0)
        {
            if (-tmp < 100*tol_)
                return 0.0;
            else
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::ConservativeSpline::sqrformula: "
                   << "invalid combination of arguments, "
                   << "attempting to extract square root of " << tmp;
                throw std::invalid_argument(os.str());
            }
        }
        return std::abs(sp)*sqrt(3*tmp);
    }

    double ConservativeSpline::aformula(const double ksq, const double s0sq,
                                        const double sp, const bool posRoot)
    {
        // Checked with Mathematica
        const double mysqrt = (posRoot ? 1 : -1)*sqrformula(ksq, s0sq, sp);
        const double sp2 = sp*sp;
        const double sp4 = sp2*sp2;
        const double s0sq2 = s0sq*s0sq;
        const double tmp = ksq*s0sq - sp2;
        return (tmp*(tmp*3*sp + mysqrt))/(18*(ksq - 1)*s0sq2*sp4);
    }

    double ConservativeSpline::rxformula(const double ksq, const double s0sq,
                                         const double sp, const bool posRoot)
    {
        // Checked with Mathematica
        const double mysqrt = (posRoot ? 1 : -1)*sqrformula(ksq, s0sq, sp);
        const double tmp = ksq*s0sq - sp*sp;
        return (tmp*3*sp - mysqrt)/2/tmp;
    }

    double ConservativeSpline::cformula(const double ksq, const double s0sq,
                                        const double sp, const bool posRoot)
    {
        // Checked with Mathematica
        const double mysqrt = (posRoot ? 1 : -1)*sqrformula(ksq, s0sq, sp);
        const double sp2 = sp*sp;
        return (sp*(ksq*s0sq + 3*sp2) + mysqrt)/(2*s0sq*sp2);
    }

    double ConservativeSpline::sqrtDerivLimitFactor(
        const double sp, const double sm)
    {
        validateSigmas("ase::ConservativeSpline::sqrtDerivLimitFactor", sp, sm);
        if (sp == sm)
            return 1.0;
        else
        {
            const double f = std::min(sp, sm)/std::max(sp, sm);
            const double f2 = f*f;
            const double onemf2 = 1.0 - f2;
            return 2*onemf2 + sqrt(4*onemf2*onemf2 + f2);
        }
    }

    std::pair<double,double> ConservativeSpline::s0sqRange(
        const double sp, const double sm, const double ksq)
    {
        const double ksqMax = sqrtDerivLimitFactor(sp, sm);
        if (ksq < 1.0 || ksq > ksqMax) throw std::invalid_argument(
            "In ase::ConservativeSpline::s0sqRange: factor out of range");
        const double sBig = std::max(sp, sm);
        const double uplim = sBig*sBig*3.0/(4.0 - 1.0/ksq);
        double lolim;
        if (ksq == ksqMax)
            lolim = uplim;
        else
        {
            const double sSmall = std::min(sp, sm);
            lolim = sSmall*sSmall*3.0/(4.0 - ksq);
            assert(lolim <= uplim);
        }
        return std::make_pair(lolim, uplim);
    }

    double ConservativeSpline::uValue(const double x) const
    {
        if (x >= rx0_)
        {
            const double ksq = getFactor(true);
            const double del = (x - sigPlus_)/sigPlus_;
            return -(ksq*del*del + c_*(x - sigPlus_) + 1)/2.0;
        }
        else if (x <= lx0_)
        {
            const double ksq = getFactor(false);
            const double del = (x + sigMinus_)/sigMinus_;
            return -(ksq*del*del + d_*(x + sigMinus_) + 1)/2.0;
        }
        else
            return -x*x*(a_*x + 1.0/s0sq_)/2.0;
    }

    double ConservativeSpline::uDerivative(const double x) const
    {
        if (x >= rx0_)
        {
            const double ksq = getFactor(true);
            return -(c_ + ksq*(2*(x - sigPlus_))/(sigPlus_*sigPlus_))/2.0;
        }
        else if (x <= lx0_)
        {
            const double ksq = getFactor(false);
            return -(d_ + ksq*(2*(x + sigMinus_))/(sigMinus_*sigMinus_))/2.0;
        }
        else
            return -x*(3.0*a_*x + 2.0/s0sq_)/2.0;
    }

    double ConservativeSpline::uSecondDerivative(const double x, double /* step */) const
    {
        if (x >= rx0_)
        {
            const double ksq = getFactor(true);
            return -ksq/(sigPlus_*sigPlus_);
        }
        else if (x <= lx0_)
        {
            const double ksq = getFactor(false);
            return -ksq/(sigMinus_*sigMinus_);
        }
        else
            return -(1.0/s0sq_ + 3.0*a_*x);
    }

    double ConservativeSpline::stepSize() const
    {
        const double tmp1 = 0.01*std::min(sigPlus_, sigMinus_);
        if (ksq_ == 1.0)
            return tmp1;
        else
        {
            const double tmp2 = std::min(std::abs(rx0_), std::abs(lx0_));
            return std::min(tmp1, tmp2);
        }
    }

    double ConservativeSpline::getFactor(const bool isRight) const
    {
        if (isRight)
            return sigPlus_ < sigMinus_ ? ksq_ : 1.0/ksq_;
        else
            return sigPlus_ < sigMinus_ ? 1.0/ksq_ : ksq_;
    }

    double ConservativeSpline::uSigmaPlus(
        const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double ConservativeSpline::uSigmaMinus(
        const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    double ConservativeSpline::unnormalizedMoment(
        const double p0, const unsigned n, const double maxDeltaLogli) const
    {
        double boundaries[4];
        const double maxArg = uArgmax();
        boundaries[0] = maxArg - uSigmaMinus(maxDeltaLogli, 1.1);
        boundaries[1] = lx0_;
        boundaries[2] = rx0_;
        boundaries[3] = maxArg + uSigmaPlus(maxDeltaLogli, 1.1);
        const GaussLegendreQuadrature glq(4U);
        const PosteriorMomentFunctor momFcn(*this, p0, n);
        const double h = stepSize();
        assert(h > 0.0);
        const double s = shift();
        long double sum = 0.0L;
        for (unsigned i=0; i<3U; ++i)
        {
            assert(boundaries[i+1] >= boundaries[i]);
            const unsigned nIntervals = (boundaries[i+1] - boundaries[i])/h + 1.0;
            sum += glq.integrate(momFcn, boundaries[i]+s, boundaries[i+1]+s, nIntervals);
        }
        return sum;
    }

    /*********************************************************************/

    MoldedCubicLogSigma::MoldedCubicLogSigma(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : DoubleCubicLogSigma(i_location, i_sigmaPlus, i_sigmaMinus,
                              getEffectiveSigmaAt0(i_sigmaPlus, i_sigmaMinus))
    {
    }

    double MoldedCubicLogSigma::getEffectiveSigmaAt0(
        const double sp, const double sm)
    {
        if (sp == sm)
            return sp;
        else
        {
            const unsigned nIntegPt = 128;
            const double tol = std::sqrt(std::numeric_limits<double>::epsilon());

            double sig0 = 0.0;
            const double guess = sqrt(moldingVarianceAt0(sp, sm, 1U));
            const MoldedCubicLogSigmaOpt opt(sp, sm, nIntegPt);
            const bool status = findMinimumGoldenSection(
                opt, std::min(sp, sm), guess, std::max(sp, sm),
                tol, &sig0);
            assert(status);
            return sig0;
        }
    }

    /*********************************************************************/

    DistributionLogli::DistributionLogli(const AbsDistributionModel1D& m,
                                         const double i_x,
                                         const double minDensity)
        : AbsShiftableLogli(0.0), distro_(m), x_(i_x)
    {
        if (!distro_.isDensityContinuous())
            throw std::invalid_argument("In ase::DistributionLogli constructor: "
                                        "inappropriate input distribution "
                                        "(density is not continuois)");
        if (!distro_.isNonNegative())
            throw std::invalid_argument("In ase::DistributionLogli constructor: "
                                        "inappropriate input distribution "
                                        "(density can be negative)");
        if (!distro_.isUnimodal())
            throw std::invalid_argument("In ase::DistributionLogli constructor: "
                                        "inappropriate input distribution "
                                        "(density is multimodal)");

        const double q0 = m.quantile(0.0);
        const double q100 = m.quantile(1.0);
        const double q16 = m.quantile(GCDF16);
        const double median = m.quantile(0.5);
        const double q84 = m.quantile(GCDF84);
        const double sp = q84 - median;
        assert(sp > 0.0);
        const double sm = median - q16;
        assert(sm > 0.0);

        pmin_ = x_ - q100;
        pmax_ = x_ - q0;
        stepSize_ = 0.01*std::min(sp, sm);
        mode_ = m.mode();
        logAtMode_ = log(m.density(mode_));

        const double cutoff = minDensity > 0.0 ? minDensity : defaultDensityCutoff_;
        adjustParameterLimits(cutoff);
    }

    void DistributionLogli::adjustParameterLimits(const double densityCutoff)
    {
        static const double tol = 2.0*DBL_EPSILON;
        static const double sqrtol = sqrt(tol);

        assert(densityCutoff > 0.0);

        const unsigned maxiter = 2000;
        const double dmax0 = pDensity(pmax_);
        const double dmin0 = pDensity(pmin_);
        if (dmax0 < densityCutoff || dmin0 < densityCutoff)
        {
            const double dmode0 = pDensity(uArgmax());
            assert(dmode0 > densityCutoff);

            if (dmax0 < densityCutoff)
            {
                double pmin = uArgmax();
                double pmax = pmax_;
                unsigned iter = 0;
                for (; iter<maxiter; ++iter)
                {
                    const double xmid = (pmin + pmax)/2.0;
                    const double fmid = pDensity(xmid);
                    if (fmid == densityCutoff)
                    {
                        pmax_ = xmid;
                        break;
                    }
                    if ((pmax - pmin)/(std::abs(xmid) + sqrtol) <= tol)
                    {
                        pmax_ = pmin;
                        break;
                    }
                    if (fmid > densityCutoff)
                        pmin = xmid;
                    else
                        pmax = xmid;
                }
                assert(iter < maxiter);
            }

            if (dmin0 < densityCutoff)
            {
                double pmin = pmin_;
                double pmax = uArgmax();
                unsigned iter = 0;
                for (; iter<maxiter; ++iter)
                {
                    const double xmid = (pmin + pmax)/2.0;
                    const double fmid = pDensity(xmid);
                    if (fmid == densityCutoff)
                    {
                        pmin_ = xmid;
                        break;
                    }
                    if ((pmax - pmin)/(std::abs(xmid) + sqrtol) <= tol)
                    {
                        pmin_ = pmax;
                        break;
                    }
                    if (fmid > densityCutoff)
                        pmax = xmid;
                    else
                        pmin = xmid;
                }
                assert(iter < maxiter);
            }
        }
    }

    double DistributionLogli::uLocation() const
    {
        return x_ - mode_;
    }

    double DistributionLogli::uArgmax() const
    {
        return x_ - mode_;
    }

    double DistributionLogli::uValue(const double p) const
    {
        if (p < pmin_ || p > pmax_) throw std::invalid_argument(
            "In ase::DistributionLogli::uValue: "
            "argument is out of range");
        return log(pDensity(p)) - logAtMode_;
    }

    double DistributionLogli::uDerivative(const double p) const
    {
        if (p < pmin_ || p > pmax_) throw std::invalid_argument(
            "In ase::DistributionLogli::uDerivative: "
            "argument is out of range");
        return pDensityDerivative(p)/pDensity(p);
    }

    double DistributionLogli::uSecondDerivative(
        const double p, const double step) const
    {
        if (p < pmin_ || p > pmax_) throw std::invalid_argument(
            "In ase::DistributionLogli::uSecondDerivative: "
            "argument is out of range");
        const double dens = pDensity(p);
        const double deriv = pDensityDerivative(p);
        const double der2 = pDensitySecondDerivative(p, step);
        const double uDer = deriv/dens;
        return der2/dens - uDer*uDer;
    }

    double DistributionLogli::pDensitySecondDerivative(
        const double p, const double step) const
    {
        const double h = step > 0.0 ? step : 0.001*stepSize_;
        double pplus = p + h;
        if (pplus > pmax_)
            pplus = pmax_;
        double pminus = p - h;
        if (pminus < pmin_)
            pminus = pmin_;
        return (pDensityDerivative(pplus) - pDensityDerivative(pminus))/
               (pplus - pminus);
    }

    double DistributionLogli::uSigmaPlus(
        const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaPlus(deltaLogli*factor(), f);
    }

    double DistributionLogli::uSigmaMinus(
        const double deltaLogli, const double f) const
    {
        return AbsLogLikelihoodCurve::sigmaMinus(deltaLogli*factor(), f);
    }

    /*********************************************************************/

    double moldingVarianceAt0(const double sp, const double sm,
                              const unsigned denomPower)
    {
        if (sp == sm)
            return sp*sp;
        else if (denomPower)
        {
            const double spDenom = pow(sp, denomPower);
            const double smDenom = pow(sm, denomPower);
            const double spNum = pow(sp, denomPower+2U);
            const double smNum = pow(sm, denomPower+2U);
            return (spNum + smNum)/(spDenom + smDenom);
        }
        else
            return (sp*sp + sm*sm)/2.0;
    }

    /*********************************************************************/

    ConservativeSigma05::ConservativeSigma05(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : ConservativeSpline(i_location, i_sigmaPlus, i_sigmaMinus,
                             conservativeSplineConstructorArgument(
                                 i_sigmaPlus, i_sigmaMinus, 1.05))
    {
    }

    ConservativeSigma10::ConservativeSigma10(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : ConservativeSpline(i_location, i_sigmaPlus, i_sigmaMinus,
                             conservativeSplineConstructorArgument(
                                 i_sigmaPlus, i_sigmaMinus, 1.1))
    {
    }

    ConservativeSigma15::ConservativeSigma15(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : ConservativeSpline(i_location, i_sigmaPlus, i_sigmaMinus,
                             conservativeSplineConstructorArgument(
                                 i_sigmaPlus, i_sigmaMinus, 1.15))
    {
    }

    ConservativeSigma20::ConservativeSigma20(const double i_location,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : ConservativeSpline(i_location, i_sigmaPlus, i_sigmaMinus,
                             conservativeSplineConstructorArgument(
                                 i_sigmaPlus, i_sigmaMinus, 1.2))
    {
    }

    ConservativeSigmaMax::ConservativeSigmaMax(const double i_location,
                                               const double i_sigmaPlus,
                                               const double i_sigmaMinus)
        : ConservativeSpline(i_location, i_sigmaPlus, i_sigmaMinus,
                             maxDerivLimitFactor(i_sigmaPlus, i_sigmaMinus))
    {
    }
}
