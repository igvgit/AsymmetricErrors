#include <cmath>
#include <cfloat>
#include <cassert>
#include <sstream>
#include <stdexcept>

#include "ase/specialFunctions.hh"
#include "ase/mathUtils.hh"
#include "ase/DistributionModels1D.hh"
#include "ase/GaussHermiteQuadrature.hh"
#include "ase/findMinimumGoldenSection.hh"

#define ONE   1.0
#define TWO   2.0
#define THREE 3.0
#define FOUR  4.0
#define DSQRT sqrt
#define LOG2PI 1.8378770664093454836
#define SQR2PI 2.5066282746310005024
#define GAUSSIAN_ENTROPY ((LOG2PI + 1.0)/2.0)

// The following function calculates correct values of beta_1 and D[beta_1,W]
// for given values of W ( = exp(delta^-2)) and beta_2 for Johnson's S_u type
// curve.
//
// Input:
//      W  - Johnson's small omega
//      B2 - beta_2
// Output:
//      B1 - beta_1
//      DB1DW - D[beta_1,w]
//      pM - Johnson's m
//
// Translated from Fortran
//
static void beta1(const long double W, const long double B2,
                  double* B1, double* DB1DW, double* pM)
{
    typedef long double Real;

    const Real WP1=W+1.0L;
    const Real WP2=W+2.0L;
    const Real WM1=W-1.0L;
    const Real B2M3=B2-3.0L;
    const Real A2 = 8.0L*(6.0L+W*(6.0L+W*(3.0L+W)));
    const Real DA2DW = 24.0L*(2.0L+W*WP2);
    const Real A1 = A2*W+8.0L*(W+3.0L);
    const Real DA1DW = DA2DW*W+A2+8.0L;
    const Real A0 = WP1*WP1*WP1*(W*W+3.0L);
    const Real DA0DW = WP1*WP1*(9.0L+W*(2.0L+5.0L*W));
    const Real A = A2*WM1-8.0L*B2M3;
    assert(A);
    const Real DADW = DA1DW-DA2DW-8.0L;
    const Real B = A1*WM1-8.0L*WP1*B2M3;
    const Real DBDW = DA1DW*WM1+A1-8.0L*B2M3;
    const Real C = A0*WM1-2.0L*B2M3*WP1*WP1;
    const Real DCDW = DA0DW*WM1+A0-4.0L*B2M3*WP1;
    const Real D = B*B-4.0L*A*C;
    assert(D > 0.0L);
    const Real DDDW = 2.0L*B*DBDW-4.0L*(A*DCDW+C*DADW);
    Real TMP = std::sqrt(D);
    Real DTMPDW = 0.5/TMP*DDDW;
    const Real M = B > 0.0L ? -2.0L*C/(B+TMP) : (TMP-B)/2.0L/A;
    const Real DMDW = (DTMPDW - DBDW - DADW*M*2.0L)/A/2.0L;

    TMP = 4.0L*WP2*M+3.0L*WP1*WP1;
    DTMPDW = 4.0L*(M+DMDW*WP2)+6.0L*WP1;
    const Real NUM = M*WM1*TMP*TMP;
    const Real DNUMDW = TMP*(WM1*(DMDW*TMP+2.0L*DTMPDW*M)+M*TMP);
    const Real loctmp = 2.0L*M+WP1;
    const Real DEN = 2.0L*loctmp*loctmp*loctmp;
    const Real DDENDW = 6.0L*loctmp*loctmp*(2.0L*DMDW+1.0L);

    *B1 = static_cast<double>(NUM/DEN);
    *DB1DW = static_cast<double>((DNUMDW*DEN-DDENDW*NUM)/(DEN*DEN));
    *pM = static_cast<double>(M);
}

static double maxent_kurtosis_below_04(const double x)
{
    static const double coeffs[8] = {
        3.0, 8.4498897194862366e-06, 2.9995305463671684,
        0.0091142952442169189, -0.52044045925140381, 0.36986923217773438,
        0.78920555114746094, -0.85073661804199219};
    return ase::polySeriesSum(coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1, x);
}

static double maxent_kurtosis_below_1(const double x)
{
    static const double xmin = 0.38, xmax = 1.0;
    static const double coeffs[10] = {
        4.5426060853567636, 1.2600472702475176, 0.14664146953014567,
        0.0023006590125657506, 0.00019877138620160838, -1.7834122791705181e-05,
        9.9733631406251355e-07, 5.4978310331382785e-08, -3.4077062090534516e-08,
        1.2460434448136049e-08};
    return ase::chebyshevSeriesSum(coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1,
                                   xmin, xmax, x);
}

static double maxent_kurtosis_below_10(const double x)
{
    static const double xmin = 0.95, xmax = 10.0;
    static const double coeffs[20] = {
        247.76242884594831, 332.45986476948843, 100.88868642905553,
        10.230982457958401, -0.29067722022231512, 0.016131389728021972,
        0.0019410656584324393, -0.0017739691088936027, 0.00084327621385860141,
        -0.00035325067750724415, 0.00014260078350414318, -5.6355831304699677e-05,
        2.2332056400120592e-05, -8.9325790053607079e-06, 3.668041834714586e-06,
        -1.3717951761194058e-06, 8.280252039583047e-07, -5.95591078678126e-08,
        1.9675452822554007e-07, -5.4652184999781639e-08};
    return ase::chebyshevSeriesSum(coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1,
                                   xmin, xmax, x);
}

static double maxent_kurtosis_below_100(const double x)
{
    static const double xmin = 9.5, xmax = 100.0;
    static const double coeffs[19] = {
        116956.30951425784, 161843.63152563464, 50644.561252902946,
        4932.9110199059569, -190.33443913205974, 24.142163899122352,
        -4.5515721784277048, 1.0631619987416343, -0.28484998461135547,
        0.083485498094887589, -0.02656100340118428, 0.0086398021594504826,
        -0.0031360156845039455, 0.00096926858896040358, -0.00045952869004395325,
        0.00012960037929587997, -4.9018872232409194e-05, 2.4740891603869386e-05,
        -1.205503212986514e-05};
    return ase::chebyshevSeriesSum(coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1,
                                   xmin, xmax, x);
}

static double maxent_kurtosis_below_730(const double x)
{
    static const double xmin = 95.0, xmax = 730.0;
    static const double coeffs[17] = {
        24656069.016257942, 33205251.773229253, 9735632.7916248273,
        860453.64801749936, -30474.25974378665, 3547.7645149184391,
        -613.62106121750548, 131.55345559783746, -32.393326121266,
        8.7459943273570389, -2.5667193054687232, 0.74322043545544147,
        -0.35008431388996542, -0.077819115947932005, -0.18331007566303015,
        -0.073552353773266077, -0.024551002657972276};
    return ase::chebyshevSeriesSum(coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1,
                                   xmin, xmax, x);
}

namespace {
    class LogCoshFcn
    {
    public:
        inline LogCoshFcn(const double gamma, const double delta)
            : gamma_(gamma), delta_(delta) {assert(delta > 0.0);}

        inline double operator()(const double x) const
            {return log(cosh((x - gamma_)/delta_));}

    private:
        double gamma_;
        double delta_;
    };

    class LogExpFcn
    {
    public:
        inline LogExpFcn(const double gamma, const double delta)
            : gamma_(gamma), delta_(delta) {assert(delta > 0.0);}

        inline double operator()(const double z) const
            {return log(1.0 + exp((z - gamma_)/delta_));}

    private:
        double gamma_;
        double delta_;
    };

    struct SuNegativeEntropy
    {
        inline SuNegativeEntropy(const double skew)
            : skew_(skew) {}

        inline double operator()(const double kurt) const
        {
            const ase::JohnsonSu distro(0.0, 1.0, skew_, kurt);
            assert(distro.isValid());
            return -distro.entropy();
        }

    private:
        double skew_;
    };
}

namespace ase {
    // The code below looks funny because it is translated from Fortran
    void JohnsonSu::initialize()
    {
        const double eps = 2.0e-13;
        const unsigned maxiter = 100000U;

        delta_ = 0.0;
        lambda_ = 0.0;
        gamma_ = 0.0;
        xi_ = 0.0;
        entropy_ = -DBL_MAX;
        entropyCalculated_ = false;

        const double B1 = skew_*skew_;
        double TMP = pow((TWO+B1+DSQRT(B1*(FOUR+B1)))/TWO, ONE/THREE);
        double W = TMP+ONE/TMP-ONE;
        TMP = W*W*(W*(W+TWO)+THREE)-THREE;
        isValid_ = kurt_ > TMP;
        if (isValid_)
        {
            // Make a guess for the value of W
            W = DSQRT(DSQRT(TWO*kurt_-TWO)-ONE)-B1/kurt_;

            // Iterations to get the correct W
            double B1TMP, DB1DW, M;
            beta1(W, kurt_, &B1TMP, &DB1DW, &M);
            unsigned count = 0U;
            while (fabs(B1-B1TMP)/(fabs(B1)+ONE) > eps && count < maxiter)
            {
                W += (B1-B1TMP)/DB1DW;
                beta1(W, kurt_, &B1TMP, &DB1DW, &M);
                ++count;
            }
            if (count >= maxiter)
            {
                // Newton-Raphson convergence is supposed to be much faster.
                // If we are here, it means we have entered an infinite loop.
                throw std::runtime_error("In ase::JohnsonSu::initialize: "
                                         "infinite loop detected");
            }
            if (M < 0.0)
                M = 0.0;
            delta_ = DSQRT(ONE/log(W));
            lambda_ = 1.0/DSQRT((W-ONE)*(TWO*M+W+ONE)/TWO);
            if (skew_)
            {
                const double sgn = skew_ > 0.0 ? 1.0 : -1.0;
                TMP = DSQRT(M/W);
                gamma_ = -sgn*fabs(delta_*log(TMP+DSQRT(TMP*TMP+ONE)));
                xi_ = -DSQRT(W)*lambda_*sgn*fabs(TMP);
            }
        }
    }

    JohnsonSu::JohnsonSu(const double location, const double scale,
                         const double skewness, const double kurtosis)
        : AbsLocationScaleFamily(location, scale),
          skew_(skewness),
          kurt_(kurtosis)
    {
        if (kurt_ == 0.0)
            kurt_ = JohnsonSystem::approxMaxEntKurtosis(skew_);
        initialize();
    }

    JohnsonSu::JohnsonSu(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1)))
    {
        assert(cumulants[1] > 0.0);
        const unsigned nCumulants = cumulants.size();
        if (nCumulants < 3U) throw std::invalid_argument(
            "In ase::JohnsonSu constructor: insufficient number of cumulants");
        skew_ = cumulants[2]/cumulants[1]/sqrt(cumulants[1]);
        if (nCumulants > 3U)
            kurt_ = cumulants[3]/cumulants[1]/cumulants[1] + 3.0;
        else
        {
            if (!skew_) throw std::invalid_argument(
                "In ase::JohnsonSu constructor: "
                "maximum extropy distribution for zero skewness "
                "is not JohnsonSu -- instead, it is the Gaussian. "
                "You probably want to construct JohnsonSystem "
                "which includes the Gaussian.");
            kurt_ = JohnsonSystem::approxMaxEntKurtosis(skew_);
        }
        initialize();
    }

    double JohnsonSu::unscaledEntropy() const
    {
        if (isValid_)
        {
            if (!entropyCalculated_)
            {
                const LogCoshFcn fcn(gamma_, delta_);
                const GaussHermiteQuadrature ghq(256);
                const double integ = ghq.integrateProb(0.0, 1.0, fcn);
                entropy_ = log(lambda_) - log(delta_) + GAUSSIAN_ENTROPY + integ;
                entropyCalculated_ = true;
            }
            return entropy_;
        }
        else
            return -DBL_MAX;
    }

    double JohnsonSu::unscaledDensity(const double x) const
    {
        if (isValid_)
        {
            const double TMP = (x-xi_)/lambda_;
            const double y = gamma_ + delta_*log(TMP+DSQRT(1.0+TMP*TMP));
            return delta_/lambda_/SQR2PI/DSQRT(1.0+TMP*TMP)*exp(-y*y/2.0);
        }
        else
            return -1.0;
    }

    double JohnsonSu::unscaledDensityDerivative(const double x) const
    {
        if (isValid_)
        {
            const double TMP = (x-xi_)/lambda_;
            const double y = gamma_ + delta_*log(TMP+DSQRT(1.0+TMP*TMP));
            const double dy = delta_/cosh(TMP);
            const double factor = delta_/lambda_/SQR2PI/lambda_;
            const double num = exp(-y*y/2.0);
            const double dnum = -y*num*dy;
            const double denom = sqrt(1.0+TMP*TMP);
            const double ddenom = TMP/denom;
            return factor*(dnum*denom - ddenom*num)/denom/denom;
        }
        else
            return 0.0;
    }

    double JohnsonSu::unscaledCdf(const double x) const
    {
        if (isValid_)
        {
            const double diff = delta_*asinh((x - xi_)/lambda_) + gamma_;
            if (diff < 0.0)
                return erfc(-diff/M_SQRT2)/2.0;
            else
                return (1.0 + erf(diff/M_SQRT2))/2.0;
        }
        else
            return -1.0;
    }

    double JohnsonSu::unscaledExceedance(const double x) const
    {
        if (isValid_)
        {
            const double diff = delta_*asinh((x - xi_)/lambda_) + gamma_;
            if (diff > 0.0)
                return erfc(diff/M_SQRT2)/2.0;
            else
                return (1.0 - erf(diff/M_SQRT2))/2.0;
        }
        else
            return -1.0;
    }

    double JohnsonSu::unscaledQuantile(const double x) const
    {
        if (!(x >= 0.0 && x <= 1.0)) throw std::domain_error(
            "In ase::JohnsonSu::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (isValid_)
            return lambda_*sinh((inverseGaussCdf(x) - gamma_)/delta_) + xi_;
        else
            return 0.0;
    }

    double JohnsonSu::unscaledInvExceedance(const double x) const
    {
        if (!(x >= 0.0 && x <= 1.0)) throw std::domain_error(
            "In ase::JohnsonSu::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (isValid_)
            return lambda_*sinh((-inverseGaussCdf(x) - gamma_)/delta_) + xi_;
        else
            return 0.0;
    }

    double JohnsonSu::unscaledRandom(AbsRNG& gen) const
    {
        if (isValid_)
        {
            const double r = Gaussian(0.0, 1.0).random(gen);
            return lambda_*sinh((r - gamma_)/delta_) + xi_;
        }
        else
            return 0.0;
    }

    double JohnsonSu::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        if (isValid_)
        {
            switch (n)
            {
            case 0U:
                cum = 1.0;
                break;
            case 1U:
                cum = 0.0;
                break;
            case 2U:
                cum = 1.0;
                break;
            case 3U:
                cum = skew_;
                break;
            case 4U:
                cum = kurt_ - 3.0;
                break;
            default:
                throw std::invalid_argument(
                    "In ase::JohnsonSu::unscaledCumulant: "
                    "only four leading cumulants are implemented");
            }
        }
        return cum;
    }

    JohnsonSb::JohnsonSb(const double location, const double scale,
                         const double skewness, const double kurtosis)
        : AbsLocationScaleFamily(location, scale),
          skew_(skewness),
          kurt_(kurtosis),
          entropy_(-DBL_MAX),
          entropyCalculated_(false)
    {
        isValid_ = fitParameters(skew_,kurt_,&gamma_,&delta_,&lambda_,&xi_);
    }

    JohnsonSb::JohnsonSb(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          kurt_(3.0),
          entropy_(-DBL_MAX),
          entropyCalculated_(false)
    {
        assert(cumulants[1] > 0.0);
        const unsigned nCumulants = cumulants.size();
        if (nCumulants < 3U) throw std::invalid_argument(
            "In ase::JohnsonSb constructor: insufficient number of cumulants");
        skew_ = cumulants[2]/cumulants[1]/sqrt(cumulants[1]);
        if (nCumulants > 3U)
            kurt_ = cumulants[3]/cumulants[1]/cumulants[1] + 3.0;
        isValid_ = fitParameters(skew_,kurt_,&gamma_,&delta_,&lambda_,&xi_);
    }

    double JohnsonSb::unscaledEntropy() const
    {
        if (isValid_)
        {
            if (!entropyCalculated_)
            {
                const LogExpFcn fcn(gamma_, delta_);
                const GaussHermiteQuadrature ghq(256);
                const double integ = ghq.integrateProb(0.0, 1.0, fcn);
                entropy_ = log(lambda_) - log(delta_) + GAUSSIAN_ENTROPY - gamma_/delta_ - 2.0*integ;
                entropyCalculated_ = true;
            }
            return entropy_;
        }
        else
            return -DBL_MAX;
    }

    double JohnsonSb::unscaledDensity(const double x) const
    {
        if (isValid_)
        {
            if (x <= xi_ || x >= xi_ + lambda_)
                return 0.0;
            else
            {
                const double y = gamma_+delta_*log((x-xi_)/(lambda_-x+xi_));
                return delta_*lambda_/
                    (exp(y*y/2.0)*SQR2PI*(x-xi_)*(lambda_-x+xi_));
            }
        }
        else
            return -1.0;
    }

    bool JohnsonSb::isUnimodal() const
    {
        if (delta_ <= 1.0/M_SQRT2)
        {
            double tmp = 1.0 - 2.0*delta_*delta_;
            if (tmp < 0.0)
                tmp = 0.0;
            tmp = sqrt(tmp);
            const double lim = tmp/delta_ - 2.0*delta_*atanh(tmp);
            if (std::abs(gamma_) <= lim)
                return false;
        }
        return true;
    }

    double JohnsonSb::unscaledDensityDerivative(const double x) const
    {
        if (isValid_)
        {
            if (x <= xi_ || x >= xi_ + lambda_)
                return 0.0;
            else
            {
                const double tmp = lambda_ - x + xi_;
                const double rat = (x-xi_)/tmp;
                const double drat = lambda_/tmp/tmp;
                const double y = gamma_ + delta_*log(rat);
                const double dy = delta_/rat*drat;
                const double factor1 = exp(y*y/2.0);
                const double df1 = y*factor1*dy;
                const double factor2 = (x-xi_)*(lambda_-x+xi_);
                const double df2 = lambda_ + 2.0*(xi_ - x);
                const double denom = factor1*factor2;
                const double ddenom = df1*factor2 + factor1*df2;
                return -delta_*lambda_*ddenom/SQR2PI/denom/denom;
            }
        }
        else
            return 0.0;
    }

    double JohnsonSb::unscaledCdf(const double x) const
    {
        if (isValid_)
        {
            if (x <= xi_)
                return 0.0;
            else if (x >= xi_ + lambda_)
                return 1.0;
            else
            {
                double diff = (x - xi_)/lambda_;
                const double tmp = diff/(1.0 - diff);
                diff = delta_*log(tmp) + gamma_;
                if (diff < 0.0)
                    return erfc(-diff/M_SQRT2)/2.0;
                else
                    return (1.0 + erf(diff/M_SQRT2))/2.0;
            }
        }
        else
            return -1.0;
    }

    double JohnsonSb::unscaledExceedance(const double x) const
    {
        if (isValid_)
        {
            if (x <= xi_)
                return 0.0;
            else if (x >= xi_ + lambda_)
                return 1.0;
            else
            {
                double diff = (x - xi_)/lambda_;
                const double tmp = diff/(1.0 - diff);
                diff = delta_*log(tmp) + gamma_;
                if (diff > 0.0)
                    return erfc(diff/M_SQRT2)/2.0;
                else
                    return (1.0 - erf(diff/M_SQRT2))/2.0;
            }
        }
        else
            return -1.0;
    }

    double JohnsonSb::unscaledQuantile(const double x) const
    {
        if (!(x >= 0.0 && x <= 1.0)) throw std::domain_error(
            "In ase::JohnsonSb::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (isValid_)
        {
            if (x == 0.0)
                return xi_;
            else if (x == 1.0)
                return xi_ + lambda_;
            else
            {
                const double tmp = exp((inverseGaussCdf(x) - gamma_)/delta_);
                return lambda_*tmp/(tmp + 1.0) + xi_;
            }
        }
        else
            return 0.0;
    }

    double JohnsonSb::unscaledInvExceedance(const double x) const
    {
        if (!(x >= 0.0 && x <= 1.0)) throw std::domain_error(
            "In ase::JohnsonSb::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (isValid_)
        {
            if (x == 1.0)
                return xi_;
            else if (x == 0.0)
                return xi_ + lambda_;
            else
            {
                const double tmp = exp((-inverseGaussCdf(x) - gamma_)/delta_);
                return lambda_*tmp/(tmp + 1.0) + xi_;
            }
        }
        else
            return 0.0;
    }

    double JohnsonSb::unscaledRandom(AbsRNG& gen) const
    {
        if (isValid_)
        {
            const double r = Gaussian(0.0, 1.0).random(gen);
            const double tmp = exp((r - gamma_)/delta_);
            return lambda_*tmp/(tmp + 1.0) + xi_;
        }
        else
            return 0.0;        
    }

    double JohnsonSb::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        if (isValid_)
        {
            switch (n)
            {
            case 0U:
                cum = 1.0;
                break;
            case 1U:
                cum = 0.0;
                break;
            case 2U:
                cum = 1.0;
                break;
            case 3U:
                cum = skew_;
                break;
            case 4U:
                cum = kurt_ - 3.0;
                break;
            default:
                throw std::invalid_argument(
                    "In ase::JohnsonSb::unscaledCumulant: "
                    "only four leading cumulants are implemented");
            }
        }
        return cum;
    }

    JohnsonSystem::JohnsonSystem(const double location, const double scale,
                                 const double skewness, const double kurtosis)
        : AbsLocationScaleFamily(location, scale),
          fcn_(0),
          skew_(skewness),
          kurt_(kurtosis)
    {
        initialize();
    }

    JohnsonSystem::JohnsonSystem(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          fcn_(0),
          skew_(0.0),
          kurt_(3.0)
    {
        assert(cumulants[1] > 0.0);
        const unsigned nCumulants = cumulants.size();
        if (nCumulants > 2U)
        {
            skew_ = cumulants[2]/cumulants[1]/sqrt(cumulants[1]);
            if (nCumulants > 3U)
                kurt_ = cumulants[3]/cumulants[1]/cumulants[1] + 3.0;
            else
                kurt_ = approxMaxEntKurtosis(skew_);
        }
        initialize();
    }

    std::string JohnsonSystem::subclass() const
    {
        switch (curveType_)
        {
        case GAUSSIAN:
            return (dynamic_cast<const Gaussian&>(*fcn_)).classname();
        case LOGNORMAL:
            return (dynamic_cast<const LogNormal&>(*fcn_)).classname();
        case SU:
            return (dynamic_cast<const JohnsonSu&>(*fcn_)).classname();
        case SB:
            return (dynamic_cast<const JohnsonSb&>(*fcn_)).classname();
        default:
            assert(0);
            return std::string();
        }
    }

    double JohnsonSystem::unscaledEntropy() const
    {
        switch (curveType_)
        {
        case GAUSSIAN:
            return (dynamic_cast<const Gaussian&>(*fcn_)).entropy();
        case LOGNORMAL:
            return (dynamic_cast<const LogNormal&>(*fcn_)).entropy();
        case SU:
            return (dynamic_cast<const JohnsonSu&>(*fcn_)).entropy();
        case SB:
            return (dynamic_cast<const JohnsonSb&>(*fcn_)).entropy();
        default:
            assert(0);
            return 0.0;
        }
    }

    double JohnsonSystem::approxMaxEntKurtosis(const double in_skew)
    {
        const double skew = std::abs(in_skew);
        if (skew < 0.38)
            return maxent_kurtosis_below_04(skew);
        else if (skew < 0.4)
        {
            const double wright = (skew - 0.38)/0.02;
            const double wleft = 1.0 - wright;
            return maxent_kurtosis_below_04(skew)*wleft +
                   maxent_kurtosis_below_1(skew)*wright;
        }
        else if (skew < 0.95)
            return maxent_kurtosis_below_1(skew);
        else if (skew < 1.0)
        {
            const double wright = (skew - 0.95)/0.05;
            const double wleft = 1.0 - wright;
            return maxent_kurtosis_below_1(skew)*wleft +
                   maxent_kurtosis_below_10(skew)*wright;
        }
        else if (skew < 9.5)
            return maxent_kurtosis_below_10(skew);
        else if (skew < 10.0)
        {
            const double wright = (skew - 9.5)/0.5;
            const double wleft = 1.0 - wright;
            return maxent_kurtosis_below_10(skew)*wleft +
                   maxent_kurtosis_below_100(skew)*wright;
        }
        else if (skew < 95.0)
            return maxent_kurtosis_below_100(skew);
        else if (skew < 100.0)
        {
            const double wright = (skew - 95.0)/5.0;
            const double wleft = 1.0 - wright;
            return maxent_kurtosis_below_100(skew)*wleft +
                   maxent_kurtosis_below_730(skew)*wright;
        }
        else if (skew <= 730.0)
            return maxent_kurtosis_below_730(skew);
        else
            return slowMaxEntKurtosis(in_skew);
    }

    double JohnsonSystem::slowMaxEntKurtosis(const double skewness)
    {
        static const double tol = 0.01*sqrt(DBL_EPSILON);

        // Here, we are assuming that the maximum entropy curve is S_u
        const double skew = std::abs(skewness);
        if (skew == 0.0)
            return 3.0;

        // First, find the kurtosis and the entropy of log-normal
        // with the same skewness. The kurtosis will be smaller than
        // the kurtosis we want and the entropy will be smaller as
        // well. Note that we are working with negative entropy here
        // (we want to minimize it).
        const SuNegativeEntropy negent(skew);
        const LogNormal lgn(0.0, 1.0, skew);
        double leftkurt = lgn.kurtosis();
        double leftne = -lgn.entropy();

        // Scan to the right and find a value of kurtosis for which
        // the negative entropy is larger than that of log-normal.
        // The exact value of the step below is not critical, we just
        // want to pick up a reasonable scale.
        //
        // Meanwhile, "bestKurt" and "bestNe" will correspond to
        // best argument and smallest value of negative entropy
        // among all calls of "negent" made so far.
        double step = skew < 5.0 ? std::max(1.0, 4.0*skew*skew) : 20.0*leftkurt;
        double rightkurt = -DBL_MAX;
        bool rightKurtFound = false;
        double bestKurt = leftkurt;
        double bestNe = leftne;
        for (unsigned i=0; i<100; ++i)
        {
            const double trykurt = leftkurt + step;
            const double tryne = negent(trykurt);
            if (tryne < bestNe)
            {
                bestNe = tryne;
                bestKurt = trykurt;
            }
            if (tryne >= leftne)
            {
                rightkurt = trykurt;
                rightKurtFound = true;
                break;
            }
            step *= 2.0;
        }
        assert(rightKurtFound);

        // Here is a problem we need to overcome. At this point,
        // our left kurtosis value can not be fed directly into
        // the minimum finder because it corresponds to a log-normal,
        // not to S_u. So, we want to find two kurtosis values
        // above log-normal kurtosis such that the negative entropy
        // is decreasing between them two.
        step = (rightkurt - leftkurt)/2.0;
        double neAtStep = negent(leftkurt + step);
        if (neAtStep < bestNe)
        {
            bestNe = neAtStep;
            bestKurt = leftkurt + step;
        }
        double halfstep = step/2.0;
        double neAtHalfStep = negent(leftkurt + halfstep);
        if (neAtHalfStep < bestNe)
        {
            bestNe = neAtHalfStep;
            bestKurt = leftkurt + halfstep;
        }
        bool leftBoundFound = false;
        for (unsigned i=0; i<100; ++i)
        {
            if (neAtStep < neAtHalfStep)
            {
                leftkurt += halfstep;
                leftBoundFound = true;
                break;
            }
            step = halfstep;
            neAtStep = neAtHalfStep;
            halfstep = step/2.0;
            neAtHalfStep = negent(leftkurt + halfstep);
            if (neAtHalfStep < bestNe)
            {
                bestKurt = leftkurt + halfstep;
                bestNe = neAtHalfStep;
            }
        }
        assert(leftBoundFound);

        // Ok, the minimum is in proper brackets and we can call the minimizer
        const bool status = findMinimumGoldenSection(
            negent, leftkurt, bestKurt, rightkurt, tol, &bestKurt);
        assert(status);
        return bestKurt;
    }

    void JohnsonSystem::initialize()
    {
        curveType_ = JohnsonSystem::select(skew_, kurt_);
        switch (curveType_)
        {
        case GAUSSIAN:
            fcn_ = new Gaussian(0.0, 1.0);
            break;
        case LOGNORMAL:
            fcn_ = new LogNormal(0.0, 1.0, skew_);
            break;
        case SU:
            {
                JohnsonSu* fu = new JohnsonSu(0.0, 1.0, skew_, kurt_);
                assert(fu->isValid());
                fcn_ = fu;
            }
            break;
        case SB:
            {
                JohnsonSb* fb = new JohnsonSb(0.0, 1.0, skew_, kurt_);
                assert(fb->isValid());
                fcn_ = fb;
            }
            break;
        case INVALID:
            {
                std::ostringstream os;
                os.precision(17);
                os << "In ase::JohnsonSystem::initialize: "
                   << "impossible combination of skewness = "
                   << skew_ << " and kurtosis = " << kurt_;
                throw std::invalid_argument(os.str());
            }
            break;
        default:
            assert(0);
        }
    }

    JohnsonSystem::JohnsonSystem(const JohnsonSystem& r)
        : AbsLocationScaleFamily(r),
          fcn_(0),
          skew_(r.skew_),
          kurt_(r.kurt_),
          curveType_(r.curveType_)
    {
        if (r.fcn_)
            fcn_ = r.fcn_->clone();
    }

    JohnsonSystem& JohnsonSystem::operator=(const JohnsonSystem& r)
    {
        if (this != &r)
        {
            AbsLocationScaleFamily* newfcn = 0;
            if (r.fcn_)
                newfcn = r.fcn_->clone();
            AbsLocationScaleFamily::operator=(r);
            skew_ = r.skew_;
            kurt_ = r.kurt_;
            curveType_ = r.curveType_;
            delete fcn_;
            fcn_ = newfcn;
        }
        return *this;
    }

    JohnsonSystem::~JohnsonSystem()
    {
        delete fcn_;
    }

    JohnsonSystem::CurveType JohnsonSystem::select(const double skew,
                                                   const double kurt)
    {
        // The tolerance here has to be made sufficiently large
        // so that the choice of Su or Sb would be correct even
        // if the calculations of the lognormal boundary were
        // performed with long double precision. This is because
        // Sb parameters are fitted using long double precision.
        static const double tol = 64.0*DBL_EPSILON;

        // Check for impossible combination of skewness and kurtosis
        const double b1 = skew*skew;
        if (kurt <= b1 + 1.0 + tol)
            return INVALID;

        // Check for Gaussian
        if (std::abs(skew) < tol && std::abs((kurt - 3.0)/3.0) < tol)
            return GAUSSIAN;

        // Check for lognormal
        const double B1 = skew*skew;
        double TMP = pow((TWO+B1+DSQRT(B1*(FOUR+B1)))/TWO, ONE/THREE);
        double W = TMP+ONE/TMP-ONE;
        TMP = W*W*(W*(W+TWO)+THREE)-THREE;
        if (std::abs((kurt - TMP)/kurt) < tol)
            return LOGNORMAL;

        // The only remaining choice is Su or Sb
        return kurt < TMP ? SB : SU;
    }
}
