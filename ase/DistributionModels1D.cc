#include <cmath>
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <sstream>

#include "ase/DistributionModels1D.hh"
#include "ase/SymmetricBetaGaussian.hh"
#include "ase/mathUtils.hh"
#include "ase/miscUtils.hh"
#include "ase/gCdfValues.hh"
#include "ase/arrayStats.hh"
#include "ase/DoubleFunctor1.hh"
#include "ase/HermiteProbOrthoPoly.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/findMinimumGoldenSection.hh"

#define LOG2PI 1.8378770664093454836
#define SQR2PI 2.5066282746310005024
#define SQR2OVERPI 0.79788456080286535588
#define GAUSSIAN_ENTROPY ((LOG2PI + 1.0)/2.0)
#define SKEW_OF_HALF_GAUSSIAN (M_SQRT2*(4.0 - M_PI)/pow(M_PI - 2.0, 1.5))

static inline int dsgn(const double v)
{
    if (v < 0.0)
        return -1;
    else if (0.0 < v)
        return 1;
    else
        return 0;
}

static inline std::pair<double,double> halfGaussSupport(
    const double sigma, const bool isTailPositive)
{
    assert(sigma > 0.0);
    if (isTailPositive)
        return std::pair<double,double>(0.0, sigma*ase::inverseGaussCdf(1.0));
    else
        return std::pair<double,double>(sigma*ase::inverseGaussCdf(0.0), 0.0);
}

static double halfGaussRandom(
    ase::AbsRNG& gen, const double sigma, const bool isTailPositive)
{
    assert(sigma > 0.0);
    const double absValue = std::abs(ase::Gaussian(0.0, sigma).random(gen));
    return isTailPositive ? absValue : -absValue;
}

// Integral from 0 to Inf (when tail is in positive direction)
// or from -Inf to 0 of (x - mu)^2 times the Gaussian density centered at 0
static double halfGaussInegral2(
    const double mu, const double sigma, const bool isTailPositive)
{
    assert(sigma > 0.0);
    const double tmp1 = (mu*mu + sigma*sigma)/2.0;
    const double tmp2 = mu*sigma*SQR2OVERPI;
    return isTailPositive ? tmp1 - tmp2 : tmp1 + tmp2;
}

static double halfGaussInegral3(
    const double mu, const double sigma, const bool isTailPositive)
{
    assert(sigma > 0.0);
    const double mu2 = mu*mu;
    const double sigma2 = sigma*sigma;
    const double tmp1 = -mu*(mu2 + 3*sigma2)/2.0;
    const double tmp2 = sigma*(3*mu2 + 2*sigma2)/SQR2PI;
    return isTailPositive ? tmp1 + tmp2 : tmp1 - tmp2;
}

static double halfGaussInegral4(
    const double mu, const double sigma, const bool isTailPositive)
{
    assert(sigma > 0.0);
    const double mu2 = mu*mu;
    const double sigma2 = sigma*sigma;
    const double mu4 = mu2*mu2;
    const double sigma4 = sigma2*sigma2;
    const double tmp1 = (mu4 + 6*mu2*sigma2 + 3*sigma4)/2.0;
    const double tmp2 = 2*mu*sigma*SQR2OVERPI*(mu2 + 2*sigma2);
    return isTailPositive ? tmp1 - tmp2 : tmp1 + tmp2;
}

namespace {
    template<class Distro>
    class QuantileSigmaRatio
    {
    public:
        inline QuantileSigmaRatio()
            : cums_(3)
        {
            cums_[0] = 0.0;
            cums_[1] = 1.0;
        }

        inline double operator()(const double skew) const
        {
            cums_[2] = skew;
            const Distro d(cums_);
            const double median = d.quantile(0.5);
            const double qplus = d.quantile(GCDF84);
            assert(qplus > median);
            const double qminus = d.quantile(GCDF16);
            assert(qminus < median);
            return (qplus - median)/(median - qminus);
        }

    private:
        mutable std::vector<double> cums_;
    };

    template<class Distro>
    class DescentDeltaRatio
    {
    public:
        inline DescentDeltaRatio(const double deltaLnL)
            : deltaLnL_(deltaLnL), cums_(3)
        {
            assert(deltaLnL_ > 0.0);
            cums_[0] = 0.0;
            cums_[1] = 1.0;
        }

        inline double operator()(const double skew) const
        {
            cums_[2] = skew;
            const Distro d(cums_);
            const double deltaPlus = d.descentDelta(true, deltaLnL_);
            assert(deltaPlus > 0.0);
            const double deltaMinus = d.descentDelta(false, deltaLnL_);
            assert(deltaMinus > 0.0);
            return deltaPlus/deltaMinus;
        }

    private:
        double deltaLnL_;
        mutable std::vector<double> cums_;
    };

    template<class Distro>
    struct QuantileSigmaRatioFromR
    {
        inline double operator()(const double r) const
        {
            assert(r > 0.0);
            const double sigmaPlus = 2.0*r/(1.0 + r);
            const double sigmaMinus = 2.0/(1.0 + r);
            const Distro d(0.0, sigmaPlus, sigmaMinus);
            const double median = d.quantile(0.5);
            const double qplus = d.quantile(GCDF84);
            assert(qplus > median);
            const double qminus = d.quantile(GCDF16);
            assert(qminus < median);
            return (qplus - median)/(median - qminus);
        }
    };

    template<class Distro>
    class DescentDeltaRatioFromR
    {
    public:
        inline DescentDeltaRatioFromR(const double deltaLnL)
            : deltaLnL_(deltaLnL) {assert(deltaLnL_ > 0.0);}

        inline double operator()(const double r) const
        {
            assert(r > 0.0);
            const double sigmaPlus = 2.0*r/(1.0 + r);
            const double sigmaMinus = 2.0/(1.0 + r);
            const Distro d(0.0, sigmaPlus, sigmaMinus);
            const double dPlus = d.descentDelta(true, deltaLnL_);
            assert(dPlus > 0.0);
            const double dMinus = d.descentDelta(false, deltaLnL_);
            assert(dMinus > 0.0);
            return dPlus/dMinus;
        }

    private:
        double deltaLnL_;
    };

    class QuantileSigmaRatioFromRSBG
    {
    public:
        inline QuantileSigmaRatioFromRSBG(const unsigned p, const double h)
            : p_(p), h_(h) {assert(h_ > 0.0);}

        inline double operator()(const double r) const
        {
            assert(r >= 0.0);
            const double sigmaPlus = 2.0*r/(1.0 + r);
            const double sigmaMinus = 2.0/(1.0 + r);
            const ase::SymmetricBetaGaussian d(0.0, sigmaPlus, sigmaMinus, p_, h_);
            const double median = d.quantile(0.5);
            const double qplus = d.quantile(GCDF84);
            assert(qplus > median);
            const double qminus = d.quantile(GCDF16);
            assert(qminus < median);
            return (qplus - median)/(median - qminus);
        }

    private:
        unsigned p_;
        double h_;
    };

    class DescentDeltaRatioFromRSBG
    {
    public:
        inline DescentDeltaRatioFromRSBG(const double deltaLnL,
                                         const unsigned p, const double h)
            : deltaLnL_(deltaLnL), h_(h), p_(p)
        {
            assert(deltaLnL_ > 0.0);
            assert(h_ > 0.0);
        }

        inline double operator()(const double r) const
        {
            assert(r >= 0.0);
            const double sigmaPlus = 2.0*r/(1.0 + r);
            const double sigmaMinus = 2.0/(1.0 + r);
            const ase::SymmetricBetaGaussian d(0.0, sigmaPlus, sigmaMinus, p_, h_);
            const double dPlus = d.descentDelta(true, deltaLnL_);
            assert(dPlus > 0.0);
            const double dMinus = d.descentDelta(false, deltaLnL_);
            assert(dMinus > 0.0);
            return dPlus/dMinus;
        }

    private:
        double deltaLnL_;
        double h_;
        unsigned p_;
    };
}

template<class Distro>
static std::unique_ptr<Distro> buildFromMedianAndSigmas(
    const double median, const double sigmaPlus, const double sigmaMinus,
    const double maxAbsSkew)
{
    assert(sigmaPlus > 0.0 && sigmaMinus > 0.0);

    double skew = 0.0;
    if (sigmaPlus != sigmaMinus)
    {
        const double r = sigmaPlus/sigmaMinus;
        double skewMin = 0.0, skewMax = 0.0;
        bool runRootFinder = true;

        if (maxAbsSkew > 0.0)
        {
            if (r > 1.0)
                skewMax = maxAbsSkew;
            else
                skewMin = -maxAbsSkew;
        }
        else
        {
            const QuantileSigmaRatio<Distro> qsr;
            double stepSize = 1.0;
            const unsigned maxSteps = 100;
            bool intervalFound = false;
            for (unsigned istep=0; istep<maxSteps && !intervalFound; ++istep)
            {
                if (r > 1.0)
                {
                    skewMax = skewMin + stepSize;
                    const double rmax = qsr(skewMax);
                    if (rmax < r)
                        skewMin = skewMax;
                    else
                    {
                        intervalFound = true;
                        if (rmax == r)
                        {
                            skew = skewMax;
                            runRootFinder = false;
                        }
                    }
                }
                else
                {
                    skewMin = skewMax - stepSize;
                    const double rmin = qsr(skewMin);
                    if (rmin > r)
                        skewMax = skewMin;
                    else
                    {
                        intervalFound = true;
                        if (rmin == r)
                        {
                            skew = skewMin;
                            runRootFinder = false;
                        }
                    }
                }
                stepSize *= 2.0;
            }
            if (!intervalFound) throw std::runtime_error(
                "In buildFromMedianAndSigmas: failed to bracket the root. "
                "Make sure that the requested sigma asymmetry is possible "
                "for this distribution.");
        }

        if (runRootFinder)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            const bool status = ase::findRootUsingBisections(
                QuantileSigmaRatio<Distro>(),
                r, skewMin, skewMax, tol, &skew);
            if (!status) throw std::runtime_error(
                "In buildFromMedianAndSigmas: root finding failed. Make sure that "
                "the requested sigma asymmetry is possible for this distribution.");
        }
    }
    std::vector<double> cums(3);
    cums[0] = 0.0;
    cums[1] = 1.0;
    cums[2] = skew;
    const Distro d(cums);
    const double med = d.quantile(0.5);
    const double qplus = d.quantile(GCDF84);
    const double qminus = d.quantile(GCDF16);
    const double scale = (sigmaPlus + sigmaMinus)/(qplus - qminus);
    assert(scale > 0.0);
    const double scalesq = scale*scale;
    cums[0] = median - med*scale;
    cums[1] = scalesq;
    cums[2] = skew*scalesq*scale;
    return std::unique_ptr<Distro>(new Distro(cums));
}

template<class Distro>
static void validateQSigmaRatio(
    const char* where, const double sigmaPlus,
    const double sigmaMinus, const double maxAbsSkew)
{
    assert(where);
    assert(sigmaPlus > 0.0 && sigmaMinus > 0.0);
    assert(maxAbsSkew > 0.0);

    if (sigmaPlus != sigmaMinus)
    {
        const double rTarget = std::max(sigmaPlus, sigmaMinus)/std::min(sigmaPlus, sigmaMinus);
        const QuantileSigmaRatio<Distro> qsr;
        const double maxDelRatio = qsr(maxAbsSkew);
        if (maxDelRatio <= rTarget)
        {
            std::ostringstream os;
            os.precision(10);
            os << "In " << where << ": "
               << "the ratio of quantile sigmas, " << rTarget << ", is too large. "
               << "It must be less than " << maxDelRatio << '.';
            throw std::invalid_argument(os.str());
        }
    }
}

template<class Distro>
static std::unique_ptr<Distro> buildFromModeAndDeltas(
    const double mode, const double sigmaPlus, const double sigmaMinus,
    const double deltaLnL, const double maxAbsSkew)
{
    assert(sigmaPlus > 0.0 && sigmaMinus > 0.0);
    assert(deltaLnL > 0.0);

    double skew = 0.0;
    if (sigmaPlus != sigmaMinus)
    {
        const double r = sigmaPlus/sigmaMinus;
        double skewMin = 0.0, skewMax = 0.0;
        bool runRootFinder = true;

        if (maxAbsSkew > 0.0)
        {
            if (r > 1.0)
                skewMax = maxAbsSkew;
            else
                skewMin = -maxAbsSkew;
        }
        else
        {
            const DescentDeltaRatio<Distro> qsr(deltaLnL);
            double stepSize = 1.0;
            const unsigned maxSteps = 100;
            bool intervalFound = false;
            for (unsigned istep=0; istep<maxSteps && !intervalFound; ++istep)
            {
                if (r > 1.0)
                {
                    skewMax = skewMin + stepSize;
                    const double rmax = qsr(skewMax);
                    if (rmax < r)
                        skewMin = skewMax;
                    else
                    {
                        intervalFound = true;
                        if (rmax == r)
                        {
                            skew = skewMax;
                            runRootFinder = false;
                        }
                    }
                }
                else
                {
                    skewMin = skewMax - stepSize;
                    const double rmin = qsr(skewMin);
                    if (rmin > r)
                        skewMax = skewMin;
                    else
                    {
                        intervalFound = true;
                        if (rmin == r)
                        {
                            skew = skewMin;
                            runRootFinder = false;
                        }
                    }
                }
                stepSize *= 2.0;
            }
            if (!intervalFound) throw std::runtime_error(
                "In buildFromModeAndDeltas: failed to bracket the root. "
                "Make sure that the requested descent delta asymmetry is "
                "possible for this distribution.");
        }

        if (runRootFinder)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            const bool status = ase::findRootUsingBisections(
                DescentDeltaRatio<Distro>(deltaLnL),
                r, skewMin, skewMax, tol, &skew);
            if (!status) throw std::runtime_error(
                "In buildFromModeAndDeltas: root finding failed. Make sure that "
                "the requested descent delta asymmetry is possible for this "
                "distribution.");
        }
    }
    std::vector<double> cums(3);
    cums[0] = 0.0;
    cums[1] = 1.0;
    cums[2] = skew;
    const Distro d(cums);
    const double unscaledMode = d.mode();
    const double delPlus = d.descentDelta(true, deltaLnL);
    const double delMinus = d.descentDelta(false, deltaLnL);
    const double scale = (sigmaPlus + sigmaMinus)/(delPlus + delMinus);
    assert(scale > 0.0);
    const double scalesq = scale*scale;
    cums[0] = mode - unscaledMode*scale;
    cums[1] = scalesq;
    cums[2] = skew*scalesq*scale;
    return std::unique_ptr<Distro>(new Distro(cums));
}

template<class Distro>
static void validateDeltasRatio(
    const char* where, const double sigmaPlus, const double sigmaMinus,
    const double deltaLnL, const double maxAbsSkew)
{
    assert(where);
    assert(sigmaPlus > 0.0 && sigmaMinus > 0.0);
    assert(deltaLnL > 0.0);
    assert(maxAbsSkew > 0.0);

    if (sigmaPlus != sigmaMinus)
    {
        const double rTarget = std::max(sigmaPlus, sigmaMinus)/std::min(sigmaPlus, sigmaMinus);
        const DescentDeltaRatio<Distro> qsr(deltaLnL);
        const double maxDelRatio = qsr(maxAbsSkew);
        if (maxDelRatio <= rTarget)
        {
            std::ostringstream os;
            os.precision(10);
            os << "In " << where << ": "
               << "the ratio of descent deltas, " << rTarget << ", is too large. "
               << "It must be less than " << maxDelRatio << '.';
            throw std::invalid_argument(os.str());
        }
    }
}

namespace ase {
    // Initialize static members.
    // The largest possible normalized skew for Fechner and Skew-normal
    // is the skew of half-Gaussian. It equals to about 0.99527174643115604244.
    const double FechnerDistribution::largestSkew_ = SKEW_OF_HALF_GAUSSIAN;
    const double SkewNormal::largestSkew_ = SKEW_OF_HALF_GAUSSIAN;

    const double QVWGaussian::largestSkew_ = 1.7722856727810639;
    const double QVWGaussian::largestAsymmetry_ = 2.0;
    const double JohnsonSystem::maxAutoSkew_ = 730.0;
    const double RailwayGaussian::maxRatioNoCutoff_ = 31.0/11.0;

    /********************************************************************/

    DimidiatedGaussian::DimidiatedGaussian(const double i_median,
                                         const double i_sigmaPlus,
                                         const double i_sigmaMinus)
        : AbsLocationScaleFamily(i_median, (std::abs(i_sigmaPlus) +
                                            std::abs(i_sigmaMinus))/2.0)
    {
        initialize(i_sigmaPlus, i_sigmaMinus);
    }

    std::unique_ptr<DimidiatedGaussian> DimidiatedGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::DimidiatedGaussian::fromQuantiles: "
            "sigma parameters must be positive");
        return std::unique_ptr<DimidiatedGaussian>(
            new DimidiatedGaussian(median, sigmaPlus, sigmaMinus));
    }

    void DimidiatedGaussian::initialize(const double i_sigmaPlus,
                                       const double i_sigmaMinus)
    {
        if (i_sigmaPlus == 0.0) throw std::invalid_argument(
            "In ase::DimidiatedGaussian::initialize: "
            "sigmaPlus parameter can not be zero");
        if (i_sigmaMinus == 0.0) throw std::invalid_argument(
            "In ase::DimidiatedGaussian::initialize: "
            "sigmaMinus parameter can not be zero");
        sigmaPlus_ = i_sigmaPlus;
        sigmaMinus_ = i_sigmaMinus;
        const double sP = std::abs(i_sigmaPlus);
        const double sM = std::abs(i_sigmaMinus);
        const double sc = (sP + sM)/2.0;
        sPstd_ = sP/sc;
        sMstd_ = sM/sc;
        const std::pair<double,double>& posSupport =
            halfGaussSupport(sPstd_, sigmaPlus_ > 0.0);
        const std::pair<double,double>& negSupport =
            halfGaussSupport(sMstd_, sigmaMinus_ < 0.0);
        xmin_ = std::min(posSupport.first, negSupport.first);
        xmax_ = std::max(posSupport.second, negSupport.second);
    }

    void DimidiatedGaussian::setScale(const double s)
    {
        if (s <= 0.0) throw std::invalid_argument(
            "In ase::DimidiatedGaussian::setScale: "
            "scale parameter must be positive");
        const double newSigmaPlus = s*sPstd_*dsgn(sigmaPlus_);
        const double newSigmaMinus = s*sMstd_*dsgn(sigmaMinus_);
        AbsLocationScaleFamily::setScale((std::abs(newSigmaPlus) + std::abs(newSigmaMinus))/2.0);
        initialize(newSigmaPlus, newSigmaMinus);
    }

    double DimidiatedGaussian::unscaledDensity(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            const double absx = std::abs(x);
            return Gaussian(0.0, sPstd_).density(absx) +
                   Gaussian(0.0, sMstd_).density(absx);
        }
        else
        {
            // Tail directions are opposite
            if (x > 0.0)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).density(x);
            }
            else if (x < 0.0)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return Gaussian(0.0, negSigma).density(x);
            }
            else
                return (Gaussian(0.0, sPstd_).density(0.0) +
                        Gaussian(0.0, sMstd_).density(0.0))/2.0;
        }
    }

    double DimidiatedGaussian::unscaledDensityDerivative(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            const double absx = std::abs(x);
            return Gaussian(0.0, sPstd_).densityDerivative(absx) +
                   Gaussian(0.0, sMstd_).densityDerivative(absx);
        }
        else
        {
            // Tail directions are opposite
            if (x > 0.0)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).densityDerivative(x);
            }
            else if (x < 0.0)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return Gaussian(0.0, negSigma).densityDerivative(x);
            }
            else
            {
                if (!(sigmaPlus_ == sigmaMinus_))
                    throw std::runtime_error(
                        "In ase::DimidiatedGaussian::unscaledDensityDerivative: "
                        "derivative at 0 is infinite");
                else
                    return 0.0;
            }
        }
    }

    double DimidiatedGaussian::unscaledCdf(const double x) const
    {
        if (x <= xmin_)
            return 0.0;
        if (x >= xmax_)
            return 1.0;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            if (sigmaPlus_ > 0.0)
                // Tail is to the right
                return Gaussian(0.0, sPstd_).cdf(x) + Gaussian(0.0, sMstd_).cdf(x) - 1.0;
            else
                // Tail is to the left
                return Gaussian(0.0, sPstd_).cdf(x) + Gaussian(0.0, sMstd_).cdf(x);
        }
        else
        {
            // Tail directions are opposite
            if (x > 0.0)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).cdf(x);
            }
            else if (x < 0.0)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return Gaussian(0.0, negSigma).cdf(x);
            }
            else
                return 0.5;
        }
    }

    double DimidiatedGaussian::unscaledExceedance(const double x) const
    {
        if (x <= xmin_)
            return 1.0;
        if (x >= xmax_)
            return 0.0;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            if (sigmaPlus_ > 0.0)
                // Tail is to the right
                return Gaussian(0.0, sPstd_).exceedance(x) + Gaussian(0.0, sMstd_).exceedance(x);
            else
                // Tail is to the left
                return 1.0 - Gaussian(0.0, sPstd_).cdf(x) - Gaussian(0.0, sMstd_).cdf(x);
        }
        else
        {
            // Tail directions are opposite
            if (x > 0.0)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).exceedance(x);
            }
            else if (x < 0.0)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return 1.0 - Gaussian(0.0, negSigma).cdf(x);
            }
            else
                return 0.5;
        }
    }

    double DimidiatedGaussian::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::DimidiatedGaussian::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return xmin_;
        if (r1 == 1.0)
            return xmax_;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledCdfFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::DimidiatedGaussian::unscaledQuantile: "
                "root finding failed");
            return q;
        }
        else
        {
            // Tail directions are opposite
            if (r1 > 0.5)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).quantile(r1);
            }
            else if (r1 < 0.5)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return Gaussian(0.0, negSigma).quantile(r1);
            }
            else
                return 0.0;
        }
    }

    double DimidiatedGaussian::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::DimidiatedGaussian::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return xmin_;
        if (r1 == 0.0)
            return xmax_;
        if (sigmaPlus_*sigmaMinus_ < 0.0)
        {
            // Tail directions of both parts are the same
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledExceedanceFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::DimidiatedGaussian::unscaledInvExceedance: "
                "root finding failed");
            return q;
        }
        else
        {
            // Tail directions are opposite
            if (r1 < 0.5)
            {
                const double posSigma = sigmaPlus_ > 0.0 ? sPstd_ : sMstd_;
                return Gaussian(0.0, posSigma).invExceedance(r1);
            }
            else if (r1 > 0.5)
            {
                const double negSigma = sigmaPlus_ > 0.0 ? sMstd_ : sPstd_;
                return Gaussian(0.0, negSigma).invExceedance(r1);
            }
            else
                return 0.0;
        }
    }

    double DimidiatedGaussian::unscaledRandom(AbsRNG& gen) const
    {
        if (gen() > 0.5)
            return halfGaussRandom(gen, sPstd_, sigmaPlus_ > 0.0);
        else
            return halfGaussRandom(gen, sMstd_, sigmaMinus_ < 0.0);
    }

    double DimidiatedGaussian::unscaledCumulant(const unsigned n) const
    {
        if (n)
        {
            const double mean = (sPstd_*dsgn(sigmaPlus_) -
                                 sMstd_*dsgn(sigmaMinus_))/SQR2PI;
            double cum = 0.0;
            switch (n)
            {
            case 1U:
                cum = mean;
                break;
            case 2U:
                cum = halfGaussInegral2(mean, sPstd_, sigmaPlus_ > 0.0) +
                      halfGaussInegral2(mean, sMstd_, sigmaMinus_ < 0.0);
                break;
            case 3U:
                cum = halfGaussInegral3(mean, sPstd_, sigmaPlus_ > 0.0) +
                      halfGaussInegral3(mean, sMstd_, sigmaMinus_ < 0.0);
                break;
            case 4U:
                {
                    const double k2 = unscaledCumulant(2U);
                    cum = halfGaussInegral4(mean, sPstd_, sigmaPlus_ > 0.0) +
                          halfGaussInegral4(mean, sMstd_, sigmaMinus_ < 0.0) - 3*k2*k2;
                }
                break;
            default:
                throw std::invalid_argument(
                    "In ase::DimidiatedGaussian::unscaledCumulant: "
                    "only four leading cumulants are implemented");
            }
            return cum;
        }
        else
            return 1.0;
    }

    double DimidiatedGaussian::skewOneTail(const double sp)
    {
        const double tmp = sp - 1;
        const double tmpsq = tmp*tmp;
        return (M_SQRT2*(4.0 + M_PI*(3.0*tmpsq - 1.0)))/pow(M_PI*(tmpsq + 1.0) - 2.0, 1.5);
    }

    double DimidiatedGaussian::skewTwoTails(const double sp)
    {
        const double tmp = sp - 1;
        const double tmpsq = tmp*tmp;
        return (M_SQRT2*tmp*((4.0 - M_PI)*tmpsq + 3.0*M_PI))/
               pow((M_PI - 2.0)*tmpsq + M_PI, 1.5);
    }

    DimidiatedGaussian::DimidiatedGaussian(const std::vector<double>& cumulants)
         : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1)))
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            static const double argLargestSkew = 1.0 - sqrt(4*M_PI*(3*M_PI - 8))/(2*M_PI);
            static const double largestSkew = 2.0/sqrt(2.0*M_PI - 5.0);
            static const double divide = (M_PI + 2.0)/pow(M_PI - 1.0, 1.5);
            static const double onem2overpi = 1.0 - 2.0/M_PI;
            static const double tol = 2.0*std::numeric_limits<double>::epsilon();

            const double absskew = std::abs(skew);
            if (absskew >= largestSkew)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::DimidiatedGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << largestSkew << '.';
                throw std::invalid_argument(os.str());
            }

            double sP = 2.0, sM = 0.0, k10 = SQR2OVERPI, k20 = onem2overpi + 1.0;
            if (absskew < divide)
            {
                // We can be either in one- or two-tail situations.
                // But here we ignore the solution with one tail.
                const bool status = findRootUsingBisections(
                    DoubleFunctor1(skewTwoTails), absskew, 0.0, 2.0, tol, &sP);
                if (!status) throw std::runtime_error(
                    "In ase::DimidiatedGaussian constructor: root finding failed (1)");
                sM = 2.0 - sP;
                const double tmp = sP - 1.0;
                k10 = tmp*SQR2OVERPI;
                k20 = onem2overpi*tmp*tmp + 1.0;
            }
            else if (absskew > divide)
            {
                // We are in the one-tail situation
                const bool status = findRootUsingBisections(
                    DoubleFunctor1(skewOneTail), absskew, 0.0, argLargestSkew, tol, &sM);
                if (!status) throw std::runtime_error(
                    "In ase::DimidiatedGaussian constructor: root finding failed (2)");
                sP = 2.0 - sM;
                sM *= -1.0;
                const double tmp = sP - 1.0;
                k20 = onem2overpi + tmp*tmp;
            }
            if (skew < 0.0)
            {
                k10 *= -1.0;
                std::swap(sP, sM);
            }
            const double scale = stdev/sqrt(k20);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - k10*scale);
            initialize(sP*scale, sM*scale);
        }
        else
            initialize(stdev, stdev);
    }

    /********************************************************************/

    std::unique_ptr<SkewNormal> SkewNormal::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::SkewNormal::fromQuantiles: "
            "sigma parameters must be positive");

        const double skewMax = largestSkew_*(1.0 - DBL_EPSILON);
        validateQSigmaRatio<SkewNormal>(
            "ase::SkewNormal::fromQuantiles",
            sigmaPlus, sigmaMinus, skewMax);

        return buildFromMedianAndSigmas<SkewNormal>(
            median, sigmaPlus, sigmaMinus, skewMax);
    }

    std::unique_ptr<SkewNormal> SkewNormal::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        // The following is the largest skew for which skew normal
        // mode finding works without problems
        static const double modeFindingSkewMax = 0.9947967308078;

        validateDeltas("ase::SkewNormal::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);
        validateDeltasRatio<SkewNormal>(
            "ase::SkewNormal::fromModeAndDeltas",
            deltaPlus, deltaMinus, deltaLnL, modeFindingSkewMax);

        return buildFromModeAndDeltas<SkewNormal>(
            mode, deltaPlus, deltaMinus, deltaLnL, modeFindingSkewMax);
    }

    SkewNormal::SkewNormal(const double i_location, const double i_scale,
                           const double i_shapeParameter)
        : AbsLocationScaleFamily(i_location, i_scale),
          alpha_(i_shapeParameter),
          rngFactor_(1.0/sqrt(1.0 + alpha_*alpha_)),
          delta_(alpha_*rngFactor_),
          g_(0.0, 1.0)
    {
        initCorrections();
    }

    void SkewNormal::initCorrections()
    {
        xmin_ = inverseGaussCdf(0.0);
        xmax_ = inverseGaussCdf(1.0);

        // These corrections are extremely small, on the order
        // of 10^(-300) or so. They are needed in order to make
        // cdf at xmin_ and exceedance at xmax_ continuous. The
        // discontinuity would constitute a problem for numerical
        // calculation of quantiles.
        cdfCorr_ = 2.0*owensT(xmin_, alpha_);
        excCorr_ = 2.0*owensT(xmax_, alpha_);
    }

    double SkewNormal::unscaledDensity(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else if (alpha_)
            return 2.0*g_.density(x)*g_.cdf(alpha_*x);
        else
            return g_.density(x);
    }

    double SkewNormal::unscaledDensityDerivative(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else if (alpha_)
            return 2.0*(g_.densityDerivative(x)*g_.cdf(alpha_*x) +
                        alpha_*g_.density(x)*g_.density(alpha_*x));
        else
            return g_.densityDerivative(x);
    }

    double SkewNormal::unscaledCdf(const double x) const
    {
        if (x <= xmin_)
            return 0.0;
        else if (x >= xmax_)
            return 1.0;
        else
        {
            double cdf = g_.cdf(x);
            if (alpha_)
                cdf -= (2.0*owensT(x, alpha_) - cdfCorr_);
            return cdf;
        }
    }

    double SkewNormal::unscaledExceedance(const double x) const
    {
        if (x <= xmin_)
            return 1.0;
        else if (x >= xmax_)
            return 0.0;
        else
        {
            double exc = g_.exceedance(x);
            if (alpha_)
                exc += (2.0*owensT(x, alpha_) - excCorr_);
            return exc;
        }
    }

    double SkewNormal::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::SkewNormal::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");

        if (r1 == 0.0)
            return xmin_;
        else if (r1 == 1.0)
            return xmax_;
        else if (alpha_)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledCdfFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::SkewNormal::unscaledQuantile: "
                "root finding failed");
            return q;
        }
        else
            return  g_.quantile(r1);
    }

    double SkewNormal::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::SkewNormal::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");

        if (r1 == 1.0)
            return xmin_;
        else if (r1 == 0.0)
            return xmax_;
        else if (alpha_)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledExceedanceFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::SkewNormal::unscaledInvExceedance: "
                "root finding failed");
            return q;
        }
        else
            return  g_.invExceedance(r1);
    }

    double SkewNormal::unscaledCumulant(const unsigned n) const
    {
        if (n)
        {
            const double mean = delta_*SQR2OVERPI;
            const double mean2 = mean*mean;
            double cum = 0.0;
            switch (n)
            {
            case 1U:
                cum = mean;
                break;
            case 2U:
                cum = 1.0 - 2.0*delta_*delta_/M_PI;
                break;
            case 3U:
                cum = (4.0 - M_PI)/2.0*mean2*mean;
                break;
            case 4U:
                cum = 2.0*(M_PI - 3.0)*mean2*mean2;
                break;
            default:
                throw std::invalid_argument(
                    "In ase::SkewNormal::unscaledCumulant: "
                    "only four leading cumulants are implemented");
            }
            return cum;
        }
        else
            return 1.0;
    }

    double SkewNormal::skewFcn(const double delta)
    {
        const double mean = delta*SQR2OVERPI;
        const double k2 = 1.0 - 2.0*delta*delta/M_PI;
        const double k3 = (4.0 - M_PI)/2.0*mean*mean*mean;
        return k3/pow(k2, 1.5);
    }

    SkewNormal::SkewNormal(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          alpha_(0.0),
          rngFactor_(1.0/sqrt(1.0 + alpha_*alpha_)),
          delta_(alpha_*rngFactor_),
          g_(0.0, 1.0)
    {
        initCorrections();
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            const double absskew = std::abs(skew);
            if (absskew >= largestSkew_)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::SkewNormal constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << largestSkew_ << '.';
                throw std::invalid_argument(os.str());
            }

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            const bool status = findRootUsingBisections(
                DoubleFunctor1(skewFcn), absskew, 0.0, 1.0, tol, &delta_);
            if (!status) throw std::runtime_error(
                "In ase::SkewNormal constructor: root finding failed");
            if (delta_ == 1.0)
                delta_ = 1.0 - std::numeric_limits<double>::epsilon();

            if (skew < 0.0)
                delta_ *= -1.0;
            rngFactor_ = sqrt(1.0 - delta_*delta_);
            alpha_ = delta_/rngFactor_;

            const double k20 = 1.0 - 2.0*delta_*delta_/M_PI;
            const double scale = stdev/sqrt(k20);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - delta_*scale*SQR2OVERPI);
        }
    }

    double SkewNormal::unscaledRandom(AbsRNG& gen) const
    {
        const double u1 = g_.random(gen);
        const double u2 = g_.random(gen);
        return delta_*std::abs(u1) + rngFactor_*u2;
    }

    /********************************************************************/

    // The following function returns
    // Integrate[x^m gcdf[x]^n g[x], {x, -Infinity, Infinity}]
    // where "g[x]" is the standard normal density and
    // "gcdf[x]" is the standard normal cumulative distribution function.
    long double QVWGaussian::moments(const unsigned m, const unsigned n)
    {
        static const unsigned mMax = 5;
        static const unsigned nMax = 5;

        static long double moms[mMax][nMax] = {
            {1.0L, 0.5L, 1.0L/3.0L, 0.25L, 0.2L},
            {0.0L, 0.2820947917738781434740397L, 0.2820947917738781434740397L, 0.2573438432509910330142467L, 0.2325928947281039225544536L},
            {1.0L, 0.5L, 0.4252214825702986749185544L, 0.3878322238554480123778316L, 0.3600040871941265652007359L},
            {0.0L, 0.7052369794346953586850993L, 0.7052369794346953586850993L, 0.6751064260945980674284983L, 0.6449758727545007761718973L},
            {3.0L, 1.5L, 1.398181980026849813535958L, 1.347272970040274720303937L, 1.304679097217132132324303L}
        };

        if (m >= mMax) throw std::invalid_argument(
            "In ase::QVWGaussian::moments: first argument is out of range");
        if (n >= nMax) throw std::invalid_argument(
            "In ase::QVWGaussian::moments: second argument is out of range");
        return moms[m][n];
    }

    double QVWGaussian::skewFcn(const double a)
    {
        const QVWGaussian qvwg(0.0, 1.0, a);
        return qvwg.unscaledCumulant(3);
    }

    void QVWGaussian::validateAsymmetry(const double a)
    {
        if (std::abs(a) >= largestAsymmetry_)
        {
            std::ostringstream os;
            os.precision(16);
            os << "In ase::QVWGaussian::validateAsymmetry: "
               << "asymmetry parameter is out of range. "
               << "|a| must be less than " << largestAsymmetry_;
            throw std::invalid_argument(os.str());
        }
    }

    void QVWGaussian::calcStandardParams(const double a)
    {
        validateAsymmetry(a);
        a_ = a;
        if (a_ == 0.0)
        {
            mu_ = 0.0;
            sigma_ = 1.0;
        }
        else
        {
            const long double m11 = moments(1U,1U);
            sigma_ = 1.0/sqrtl(1.0 + a_*a_*(moments(2U,2U) - 0.25L - m11*m11));
            mu_ = -a_*m11*sigma_;
        }
    }

    QVWGaussian::QVWGaussian(const double i_location, const double i_scale,
                             const double i_asymmetryParameter)
        : AbsLocationScaleFamily(i_location, i_scale),
          g_(0.0, 1.0)
    {
        calcStandardParams(i_asymmetryParameter);
    }

    QVWGaussian::QVWGaussian(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          g_(0.0, 1.0)
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        double a = 0.0;
        if (skew)
        {
            if (std::abs(skew) >= largestSkew_)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::QVWGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << largestSkew_ << '.';
                throw std::invalid_argument(os.str());
            }

            const double maxa = largestAsymmetry_*(1.0 - DBL_EPSILON);
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            const bool status = findRootUsingBisections(
                DoubleFunctor1(skewFcn), skew, -maxa, maxa, tol, &a);
            if (!status) throw std::runtime_error(
                "In ase::QVWGaussian constructor: root finding failed");
        }
        calcStandardParams(a);
    }

    double QVWGaussian::unscaledRandom(AbsRNG& gen) const
    {
        const double u = g_.random(gen);
        if (a_)
        {
            const double y = g_.cdf(u);
            return mu_ + sigma_*(1.0 + a_*(y - 0.5))*u;
        }
        else
            return u;
    }

    double QVWGaussian::unscaledDensity(const double x0) const
    {
        if (a_)
        {
            const double y = unscaledCdf(x0);
            const double x = g_.quantile(y);
            const double gd = g_.density(x);
            if (gd > 0.0)
            {
                const double tmp = sigma_*(1.0 + a_*(y - 0.5))/gd + x*sigma_*a_;
                assert(tmp > 0.0);
                return 1.0/tmp;
            }
            else
                return 0.0;
        }
        else
            return g_.density(x0);
    }

    double QVWGaussian::unscaledDensityDerivative(const double x0) const
    {
        if (a_)
        {
            const double y = unscaledCdf(x0);
            const double x = g_.quantile(y);
            const double gd = g_.density(x);
            if (gd > 0.0)
            {
                const double tmp = sigma_*(1.0 + a_*(y - 0.5))/gd + x*sigma_*a_;
                assert(tmp > 0.0);
                const double dens = 1.0/tmp;
                const double dgd = -g_.densityDerivative(x)/gd/gd/gd;
                return -dens*dens*dens*(sigma_*(1.0 + a_*(y - 0.5))*dgd + 2.0*sigma_*a_/gd);
            }
            else
                return 0.0;
        }
        else
            return g_.densityDerivative(x0);
    }

    double QVWGaussian::unscaledExceedance(const double x) const
    {
        return 1.0 - unscaledCdf(x);
    }

    double QVWGaussian::unscaledCdf(const double x) const
    {
        if (a_)
        {
            const double xmin = unscaledQuantile(0.0);
            if (x <= xmin)
                return 0.0;
            const double xmax = unscaledQuantile(1.0);
            if (x >= xmax)
                return 1.0;

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double y;
            const bool status = findRootUsingBisections(
                UnscaledQuantileFunctor1D(*this), x, 0.0, 1.0, tol, &y);
            if (!status) throw std::runtime_error(
                "In ase::QVWGaussian::unscaledCdf: root finding failed");
            return y;
        }
        else
            return g_.cdf(x);
    }

    double QVWGaussian::unscaledQuantile(const double r1) const
    {
       if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::QVWGaussian::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
       return mu_ + sigma_*(1.0 + a_*(r1 - 0.5))*g_.quantile(r1);
    }

    double QVWGaussian::unscaledInvExceedance(const double r1) const
    {
       if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::QVWGaussian::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
       return mu_ + sigma_*(1.0 + a_*(0.5 - r1))*g_.invExceedance(r1);
    }

    double QVWGaussian::unscaledCumulant(const unsigned n) const
    {
        // The mean and the variance of the unscaled density
        // are standardized
        if (n == 0U || n == 2U)
            return 1.0;
        else if (n == 1U)
            return 0.0;
        else if (a_)
        {
            const long double a = a_;
            const long double a2 = a*a;
            const long double tmp = a - 2.0L;
            const long double tmp2 = tmp*tmp;
            const long double sigma = sigma_;
            const long double sigma2 = sigma*sigma;
            const long double sigma3 = sigma2*sigma;
            const long double mom11 = moments(1, 1);
            const long double mom112 = mom11*mom11;
            const long double mom113 = mom112*mom11;
            double cum;

            switch (n)
            {
            case 3U:
                 cum = (a*sigma3*(8*a2*mom113 - 3*mom11*(4 + a2*(-1 + 4*moments(2, 2))) + 
                                  3*tmp2*moments(3, 1) - 6*tmp*a*moments(3, 2) + 4*a2*moments(3, 3)))/4;
                 break;
            case 4U:
            {
                const long double sigma4 = sigma2*sigma2;
                const long double mom114 = mom112*mom112;
                const long double a3 = a*a2;
                const long double a4 = a2*a2;

                cum = -(sigma4*(-48 - 24*a2*(-3 + 4*mom112 - 8*mom11*moments(3, 1) + 
                       4*moments(4, 2)) - 16*a3*(3 + 12*mom11*(moments(3, 1) - moments(3, 2)) - 
                       6*moments(4, 2) + 4*moments(4, 3)) + a4*(9 + 48*mom114 + 
                       mom112*(24 - 96*moments(2, 2)) + 16*mom11*
                       (3*moments(3, 1) - 6*moments(3, 2) + 4*moments(3, 3)) - 24*moments(4, 2) + 
                       32*moments(4, 3) - 16*moments(4, 4))))/16 - 3.0;
            }
            break;
            default:
                throw std::invalid_argument(
                    "In ase::QVWGaussian::unscaledCumulant: "
                    "only four leading cumulants are implemented");
            }
            return cum;
        }
        else
            return 0.0;
    }

    std::unique_ptr<QVWGaussian> QVWGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::QVWGaussian::fromQuantiles: "
            "sigma parameters must be positive");

        const double skewMax = largestSkew_*(1.0 - DBL_EPSILON);
        validateQSigmaRatio<QVWGaussian>(
            "ase::QVWGaussian::fromQuantiles",
            sigmaPlus, sigmaMinus, skewMax);

        return buildFromMedianAndSigmas<QVWGaussian>(
            median, sigmaPlus, sigmaMinus, skewMax);
    }

    std::unique_ptr<QVWGaussian> QVWGaussian::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        // The ratio of deltas (calculated at deltaLnL = 0.5)
        // starts decreasing above the following skewness
        const double maxInputSkew = 1.3102319667389;

        validateDeltas("ase::QVWGaussian::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);
        validateDeltasRatio<QVWGaussian>(
            "ase::QVWGaussian::fromModeAndDeltas",
            deltaPlus, deltaMinus, deltaLnL, maxInputSkew);

        return buildFromModeAndDeltas<QVWGaussian>(
            mode, deltaPlus, deltaMinus, deltaLnL, maxInputSkew);
    }

    /********************************************************************/

    void GammaDistribution::initialize()
    {
        if (!(alpha_ > 0.0)) throw std::invalid_argument(
            "In ase::GammaDistribution::initialize: invalid shape parameter");
        norm_ = 1.0/Gamma(alpha_);
        uplim_ = -log(std::numeric_limits<double>::min());
    }

    GammaDistribution::GammaDistribution(const double location,
                                         const double scale,
                                         const double shapeParameter)
        :  AbsLocationScaleFamily(location, scale),
           alpha_(shapeParameter)
    {
        initialize();
    }

    GammaDistribution::GammaDistribution(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(0.0, 1.0)
    {
        const unsigned nCumulants = cumulants.size();
        if (nCumulants < 3U) throw std::invalid_argument(
            "In ase::GammaDistribution constructor: insufficient number of cumulants");
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double k3 = cumulants[2];
        if (!(k3 > 0.0)) throw std::invalid_argument(
            "In ase::GammaDistribution constructor: third cumulant must be positive");
        const double s = k3/k2/2.0;
        alpha_ = k2/s/s;
        AbsLocationScaleFamily::setScale(s);
        AbsLocationScaleFamily::setLocation(cumulants[0] - alpha_*s);
        initialize();
    }

    double GammaDistribution::unscaledDensity(const double x) const
    {
        if (x > 0.0 && x <= uplim_)
            return norm_*pow(x, alpha_-1.0)*exp(-x);
        else
            return 0.0;
    }

    double GammaDistribution::unscaledDensityDerivative(const double x) const
    {
        if (x > 0.0 && x <= uplim_)
            return norm_*pow(x, alpha_-2.0)*exp(-x)*(alpha_ - 1.0 - x);
        else
            return 0.0;
    }

    double GammaDistribution::unscaledCdf(const double x) const
    {
        if (x <= 0.0)
            return 0.0;
        else if (x >= uplim_)
            return 1.0;
        else
            return incompleteGamma(alpha_, x);
    }

    double GammaDistribution::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::GammaDistribution::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return 0.0;
        else if (r1 == 1.0)
            return uplim_;
        else
            return inverseIncompleteGamma(alpha_, r1);
    }

    double GammaDistribution::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::GammaDistribution::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return 0.0;
        else if (r1 == 0.0)
            return uplim_;
        else
            return inverseIncompleteGammaC(alpha_, r1);
    }

    double GammaDistribution::unscaledExceedance(const double x) const
    {
        if (x <= 0.0)
            return 1.0;
        else if (x >= uplim_)
            return 0.0;
        else
            return incompleteGammaC(alpha_, x);
    }

    double GammaDistribution::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        switch (n)
        {
        case 0U:
            cum = 1.0;
            break;
        case 1U:
            cum = alpha_;
            break;
        case 2U:
            cum = alpha_;
            break;
        case 3U:
            cum = 2.0*alpha_;
            break;
        case 4U:
            cum = 6.0*alpha_;
            break;
        default:
            throw std::invalid_argument(
                "In ase::GammaDistribution::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    // Using Marsaglia and Tsang method
    double GammaDistribution::unscaledRnd(const double alpha, AbsRNG& gen)
    {
        if (alpha > 1.0)
        {
            const Gaussian g(0.0, 1.0);

            const double d = alpha - 1.0/3.0;
            const double c = 1.0/sqrt(9.0*d);
            const double moneovc = -1.0/c;

            double v = 0.0;
            bool cond = true;
            while (cond)
            {
                const double z = g.random(gen);
                if (z > moneovc)
                {
                    const double tmp = (1.0 + c*z);
                    v = tmp*tmp*tmp;
                    cond = log(gen()) > (0.5*z*z + d - d*v + d*log(v));
                }
            }
            return d*v;
        }
        else if (alpha == 1.0)
            // This distribution is just exponential
            return -log(1.0 - gen());
        else
            return unscaledRnd(alpha + 1.0, gen)*pow(gen(), 1.0/alpha);
    }

    /********************************************************************/

    std::unique_ptr<LogNormal> LogNormal::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::LogNormal::fromQuantiles: "
            "sigma parameters must be positive");
        return buildFromMedianAndSigmas<LogNormal>(
            median, sigmaPlus, sigmaMinus, 0.0);
    }

    std::unique_ptr<LogNormal> LogNormal::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        // The following is the largest skew for which skew normal
        // mode finding works without problems
        static const double modeFindingSkewMax = 974051368.769;

        validateDeltas("ase::LogNormal::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);
        validateDeltasRatio<LogNormal>(
            "ase::LogNormal::fromModeAndDeltas",
            deltaPlus, deltaMinus, deltaLnL, modeFindingSkewMax);

        return buildFromModeAndDeltas<LogNormal>(
            mode, deltaPlus, deltaMinus, deltaLnL, modeFindingSkewMax);
    }

    LogNormal::LogNormal(const double mean, const double stdev,
                         const double i_skewness)
        : AbsLocationScaleFamily(mean, stdev),
          skew_(i_skewness)
    {
        initialize();
    }

    LogNormal::LogNormal(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          skew_(0.0)
    {
        assert(cumulants[1] > 0.0);
        if (cumulants.size() > 2U)
        {
            const double sigma = sqrt(cumulants[1]);
            skew_ = cumulants[2]/cumulants[1]/sigma;
        }
        initialize();
    }

    void LogNormal::initialize()
    {
        w_ = 1.0;
        logw_ = 0.0;
        s_ = 0.0;
        xi_ = 0.0;
        emgamovd_ = 0.0;

        if (skew_)
        {
            const double b1 = skew_*skew_;
            const double tmp = pow((2.0+b1+sqrt(b1*(4.0+b1)))/2.0, 1.0/3.0);
            w_ = tmp + 1.0/tmp - 1.0;
            logw_ = log(w_);
            if (logw_ > 0.0)
            {
                s_ = sqrt(logw_);
                emgamovd_ = 1.0/sqrt(w_*(w_-1.0));
                xi_ = -emgamovd_*sqrt(w_);
            }
            else
            {
                // This is not different from a Gaussian within
                // the numerical precision of our calculations
                w_ = 1.0;
                logw_ = 0.0;
                skew_ = 0.0;
            }
        }
    }

    double LogNormal::unscaledMode() const
    {
        if (skew_)
        {
            const double tmp = emgamovd_/exp(logw_) + xi_;
            return skew_ > 0.0 ? tmp : -tmp;
        }
        else
            return 0.0;
    }

    double LogNormal::unscaledEntropy() const
    {
        if (skew_)
        {
            const double mu = log(emgamovd_);
            return log(s_) + mu + GAUSSIAN_ENTROPY;
        }
        else
        {
            // This is a Gaussian
            return GAUSSIAN_ENTROPY;
        }
    }

    double LogNormal::unscaledDensity(const double x) const
    {
        if (skew_)
        {
            const double diff = skew_ > 0.0 ? x - xi_ : -x - xi_;
            if (diff <= 0.0)
                return 0.0;
            else
            {
                const double lg = log(diff/emgamovd_);
                return exp(-lg*lg/2.0/logw_)/s_/SQR2PI/diff;
            }
        }
        else
        {
            // This is a Gaussian
            return exp(-x*x/2.0)/SQR2PI;
        }
    }

    double LogNormal::unscaledDensityDerivative(const double x) const
    {
        if (skew_)
        {
            const double diff = skew_ > 0.0 ? x - xi_ : -x - xi_;
            if (diff <= 0.0)
                return 0.0;
            else
            {
                const double diffDer = skew_ > 0.0 ? 1.0 : -1.0;
                const double lg = log(diff/emgamovd_);
                const double lgDer = diffDer/diff;
                const double num = exp(-lg*lg/2.0/logw_);
                const double numDer = -lg*num/logw_*lgDer;

                return (numDer*diff - diffDer*num)/diff/diff/s_/SQR2PI;
            }
        }
        else
        {
            // This is a Gaussian
            return -x*exp(-x*x/2.0)/SQR2PI;
        }
    }

    double LogNormal::unscaledCdf(const double x) const
    {
        if (skew_)
        {
            const double diff = skew_ > 0.0 ? x - xi_ : -x - xi_;
            double posCdf = 0.0;
            if (diff > 0.0)
                posCdf = (1.0 + erf(log(diff/emgamovd_)/s_/M_SQRT2))/2.0;
            return skew_ > 0.0 ? posCdf : 1.0 - posCdf;
        }
        else
            return (1.0 + erf(x/M_SQRT2))/2.0;
    }

    double LogNormal::unscaledExceedance(const double x) const
    {
        // Some day we should fix this...
        return 1.0 - unscaledCdf(x);
    }

    double LogNormal::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::LogNormal::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        const double g = inverseGaussCdf(skew_ >= 0.0 ? r1 : 1.0 - r1);
        if (skew_)
        {
            const double v = emgamovd_*exp(s_*g) + xi_;
            return skew_ > 0.0 ? v : -v;
        }
        else
            return g;
    }

    double LogNormal::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::LogNormal::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        const double g = -inverseGaussCdf(skew_ >= 0.0 ? r1 : 1.0 - r1);
        if (skew_)
        {
            const double v = emgamovd_*exp(s_*g) + xi_;
            return skew_ > 0.0 ? v : -v;
        }
        else
            return g;
    }

    double LogNormal::kurtosis() const
    {
        return w_*w_*(w_*(w_+2.0)+3.0)-3.0;
    }

    double LogNormal::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
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
            cum = kurtosis() - 3.0;
            break;
        default:
            throw std::invalid_argument(
                "In ase::LogNormal::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    double LogNormal::unscaledRandom(AbsRNG& gen) const
    {
        const double g = Gaussian(0.0, 1.0).random(gen);
        if (skew_)
        {
            const double v = emgamovd_*exp(s_*g) + xi_;
            return skew_ > 0.0 ? v : -v;
        }
        else
            return g;
    }

    /********************************************************************/

    std::unique_ptr<DistortedGaussian> DistortedGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::DistortedGaussian::fromQuantiles: "
            "sigma parameters must be positive");

        const double skewMax = 2.0*M_SQRT2*(1.0 - DBL_EPSILON);
        validateQSigmaRatio<DistortedGaussian>(
            "ase::DistortedGaussian::fromQuantiles",
            sigmaPlus, sigmaMinus, skewMax);

        return buildFromMedianAndSigmas<DistortedGaussian>(
            median, sigmaPlus, sigmaMinus, skewMax);
    }

    void DistortedGaussian::initialize(const double i_sigmaPlus,
                                       const double i_sigmaMinus)
    {
        if (i_sigmaPlus == 0.0 && i_sigmaMinus == 0.0)
            throw std::invalid_argument(
                "In ase::DistortedGaussian::initialize: "
                "both scale parameters can not be zero");
        sigmaPlus_ = i_sigmaPlus;
        sigmaMinus_ = i_sigmaMinus;
        const double sP = std::abs(i_sigmaPlus);
        const double sM = std::abs(i_sigmaMinus);
        const double sc = (sP + sM)/2.0;
        sPstd_ = i_sigmaPlus/sc;
        sMstd_ = i_sigmaMinus/sc;
        alpha_ = (sPstd_ - sMstd_)/2.0;
        sigma_ = (sPstd_ + sMstd_)/2.0;
        xmin_ = inverseGaussCdf(0.0);
        xmax_ = inverseGaussCdf(1.0);
        if (alpha_)
        {
            const double extremum = -sigma_*sigma_/4.0/alpha_;
            if (alpha_ > 0.0)
                xmin_ = std::max(xmin_, extremum);
            else if (alpha_ < 0.0)
                xmax_ = std::min(xmax_, extremum);
        }
    }

    DistortedGaussian::DistortedGaussian(const double i_location,
                                         const double i_sigmaPlus,
                                         const double i_sigmaMinus)
        : AbsLocationScaleFamily(i_location, (std::abs(i_sigmaPlus) +
                                              std::abs(i_sigmaMinus))/2.0),
          g_(0.0, 1.0)
    {
        initialize(i_sigmaPlus, i_sigmaMinus);
    }

    void DistortedGaussian::setScale(const double s)
    {
        if (s <= 0.0) throw std::invalid_argument(
            "In ase::DistortedGaussian::setScale: "
            "scale parameter must be positive");
        const double newSigmaPlus = s*sPstd_;
        const double newSigmaMinus = s*sMstd_;
        AbsLocationScaleFamily::setScale((std::abs(newSigmaPlus) + std::abs(newSigmaMinus))/2.0);
        initialize(newSigmaPlus, newSigmaMinus);
    }

    void DistortedGaussian::findAndValidateRoots(
        const double x, double* u1, double* u2) const
    {
        assert(u1);
        assert(u2);
        assert(alpha_);

        const double b = sigma_/alpha_;
        const double c = -x/alpha_;
        const unsigned nRoots = solveQuadratic(b, c, u1, u2);
        if (nRoots != 2U)
        {
            // In principle, we should never be in this branch,
            // but round-off errors are nasty critters...
            std::ostringstream os;
            os.precision(16);
            os << "In ase::DistortedGaussian::validateRoots: "
               << "failed to find the roots of the quadratic equation for "
               << "x = " << x
               << ", alpha = " << alpha_
               << ", sigma = " << sigma_
               << ", xmin = " << xmin_
               << ", xmax = " << xmax_;
            throw std::runtime_error(os.str());
        }
        if (*u1 > *u2)
            std::swap(*u1, *u2);
    }

    double DistortedGaussian::unscaledDensity(const double x) const
    {
        if (alpha_)
        {
            if (x <= xmin_ || x >= xmax_)
                return 0.0;
            double u1, u2;
            findAndValidateRoots(x, &u1, &u2);
            return g_.density(u1)/std::abs(sigma_ + 2*alpha_*u1) +
                   g_.density(u2)/std::abs(sigma_ + 2*alpha_*u2);
        }
        else
            return g_.density(x);
    }

    double DistortedGaussian::unscaledDensityDerivative(const double x) const
    {
        if (alpha_)
        {
            if (x <= xmin_ || x >= xmax_)
                return 0.0;
            double uvals[2];
            findAndValidateRoots(x, &uvals[0], &uvals[1]);
            long double sum = 0.0L;
            for (unsigned i=0; i<2U; ++i)
            {
                const double u = uvals[i];
                const double der = sigma_ + 2*alpha_*u;
                const double absder = std::abs(der);
                const double sder = 2*alpha_;
                const double dens = g_.density(u);

                sum += (g_.densityDerivative(u) - dens*sder/der)/der/absder;
            }
            return sum;
        }
        else
            return g_.densityDerivative(x);
    }

    double DistortedGaussian::unscaledRandom(AbsRNG& gen) const
    {
        const double x = g_.random(gen);
        return alpha_*x*x + sigma_*x;
    }

    double DistortedGaussian::unscaledCdf(const double x) const
    {
        if (x <= xmin_)
            return 0.0;
        if (x >= xmax_)
            return 1.0;

        if (alpha_)
        {
            double u1, u2;
            findAndValidateRoots(x, &u1, &u2);
            if (alpha_ > 0.0)
                return g_.cdf(u2) - g_.cdf(u1);
            else
                return g_.cdf(u1) + g_.exceedance(u2);
        }
        else
            return g_.cdf(x);
    }

    double DistortedGaussian::unscaledExceedance(const double x) const
    {
        if (x <= xmin_)
            return 1.0;
        if (x >= xmax_)
            return 0.0;

        if (alpha_)
        {
            double u1, u2;
            findAndValidateRoots(x, &u1, &u2);
            if (alpha_ > 0.0)
                return g_.cdf(u1) + g_.exceedance(u2);
            else
                return g_.cdf(u2) - g_.cdf(u1);
        }
        else
            return g_.exceedance(x);
    }

    double DistortedGaussian::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::DistortedGaussian::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return xmin_;
        if (r1 == 1.0)
            return xmax_;

        if (alpha_)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledCdfFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::DistortedGaussian::unscaledQuantile: "
                "root finding failed");
            return q;
        }
        else
            return g_.quantile(r1);
    }

    double DistortedGaussian::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::DistortedGaussian::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return xmin_;
        if (r1 == 0.0)
            return xmax_;

        if (alpha_)
        {
            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double q;
            const bool status = findRootUsingBisections(
                UnscaledExceedanceFunctor1D(*this), r1, xmin_, xmax_, tol, &q);
            if (!status) throw std::runtime_error(
                "In ase::DistortedGaussian::unscaledInvExceedance: "
                "root finding failed");
            return q;
        }
        else
            return g_.invExceedance(r1);
    }

    double DistortedGaussian::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        switch (n)
        {
        case 0U:
            cum = 1.0;
            break;
        case 1U:
            cum = alpha_;
            break;
        case 2U:
            cum = 2.0*alpha_*alpha_ + sigma_*sigma_;
            break;
        case 3U:
            cum = 2.0*alpha_*(4.0*alpha_*alpha_ + 3.0*sigma_*sigma_);
            break;
        case 4U:
            cum = 48.0*alpha_*alpha_*(alpha_*alpha_ + sigma_*sigma_);
            break;
        default:
            throw std::invalid_argument(
                "In ase::DistortedGaussian::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    double DistortedGaussian::skewAlphaOneTail(const double sigma)
    {
        const double k2 = 2.0 + sigma*sigma;
        const double stdev = sqrt(k2);
        return 2.0*(4.0 + 3.0*sigma*sigma)/k2/stdev;
    }

    double DistortedGaussian::skewAlphaTwoTails(const double alpha)
    {
        const double k2 = 2.0*alpha*alpha + 1.0;
        const double stdev = sqrt(k2);
        return 2.0*alpha*(2*k2 + 1.0)/k2/stdev;
    }

    DistortedGaussian::DistortedGaussian(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          g_(0.0, 1.0)
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            static const double sqrt8 = 2.0*M_SQRT2;
            static const double divide = 14.0/(3.0*sqrt(3.0));
            static const double tol = 2.0*std::numeric_limits<double>::epsilon();

            // For distorted Gaussian, the largest possible absolute
            // skewness is sqrt(8) = 2.82843... The skewness at alpha = 1
            // is 14/(3 sqrt(3)) = 2.6943... For skewness values in between
            // these two, we are in the realm of one-tailed distributions.
            const double absskew = std::abs(skew);
            if (absskew >= sqrt8)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::DistortedGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << sqrt8 << '.';
                throw std::invalid_argument(os.str());
            }

            double absalpha = 1.0, sigma = 1.0;
            if (absskew < divide)
            {
                const bool status = findRootUsingBisections(
                    DoubleFunctor1(skewAlphaTwoTails), absskew, 0.0, 1.0, tol, &absalpha);
                if (!status) throw std::runtime_error(
                    "In ase::DistortedGaussian constructor: root finding failed (1)");
            }
            else if (absskew > divide)
            {
                const bool status = findRootUsingBisections(
                    DoubleFunctor1(skewAlphaOneTail), absskew, 0.0, 1.0, tol, &sigma);
                if (!status) throw std::runtime_error(
                    "In ase::DistortedGaussian constructor: root finding failed (2)");
            }
            const double alpha = absalpha*dsgn(skew);
            const double scale = stdev/sqrt(2.0*alpha*alpha + sigma*sigma);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - alpha*scale);
            double sP, sM;
            if (absskew <= divide)
            {
                sP = 1.0 + alpha;
                sM = 1.0 - alpha;
            }
            else
            {
                sP = sigma + 1.0;
                sM = sigma - 1.0;
                if (skew < 0.0)
                    std::swap(sP, sM);
            }
            initialize(sP*scale, sM*scale);
        }
        else
            initialize(stdev, stdev);
    }

    /********************************************************************/

    std::unique_ptr<RailwayGaussian> RailwayGaussian::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        validateDeltas("ase::RailwayGaussian::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);

        if (deltaPlus == deltaMinus)
        {
            const double sig = deltaPlus/sqrt(2.0*deltaLnL);
            return std::unique_ptr<RailwayGaussian>(
                new RailwayGaussian(mode, sig, sig));
        }

        // The following cutoff is not set at maxRatioNoCutoff_ because
        // the ratio of descent deltas as a function of the ratio of
        // input sigmas changes its behavior from increasing to decreasing
        // at "maxInputRatio" (this number is calculated here for deltaLnL = 0.5).
        // Note that this number must be consistent with "minH" and "maxH"
        // definitions inside "transitionRegionChoice".
        const double maxInputRatio = 1.84390478347;
        const DescentDeltaRatioFromR<RailwayGaussian> delRatio(deltaLnL);
        const double maxDelRatio = delRatio(maxInputRatio);
        const double rbig = std::max(deltaPlus, deltaMinus)/std::min(deltaPlus, deltaMinus);
        if (rbig >= maxDelRatio)
        {
            std::ostringstream os;
            os.precision(10);
            os << "In ase::RailwayGaussian::fromModeAndDeltas: "
               << "the ratio of descent deltas, " << rbig << ", is too large. "
               << "It must be less than " << maxDelRatio << '.';
            throw std::invalid_argument(os.str());
        }
        double inpR;
        const bool status = findRootUsingBisections(
            delRatio, rbig, 1.0, maxInputRatio,
            2.0*std::numeric_limits<double>::epsilon(), &inpR);
        assert(status);
        double sigmaP0 = 2.0*inpR/(1.0 + inpR);
        double sigmaM0 = 2.0/(1.0 + inpR);
        if (deltaPlus/deltaMinus < 1.0)
            std::swap(sigmaP0, sigmaM0);
        const RailwayGaussian distro(0.0, sigmaP0, sigmaM0);
        const double delPlus = distro.descentDelta(true, deltaLnL);
        const double delMinus = distro.descentDelta(false, deltaLnL);
        const double scale = (deltaPlus + deltaMinus)/(delPlus + delMinus);
        assert(scale > 0.0);
        const double unscaledMode = distro.mode();
        const double newMed = mode - unscaledMode*scale;
        return std::unique_ptr<RailwayGaussian>(new RailwayGaussian(
            newMed, sigmaP0*scale, sigmaM0*scale));
    }

    std::unique_ptr<RailwayGaussian> RailwayGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        // The following constants must be in agreement with
        // "minH" and "maxH" definitions inside "transitionRegionChoice".
        //
        // The largest possible quantile r needs to be determined numerically.
        // To be on the safe side, we will limit the input r by the number
        // which is a tiny bit smaller. At the same time, the corresponding
        // input r will be almost strictly at the maximum.
        static const double maxQuantileR = 4.059106719517114;
        static const double maxInputR = 3.971005;

        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::RailwayGaussian::fromQuantiles: "
            "sigma parameters must be positive");
        const double quantileR = sigmaPlus/sigmaMinus;
        if (quantileR > maxQuantileR || quantileR < 1.0/maxQuantileR)
            throw std::invalid_argument(
                "In ase::RailwayGaussian::fromQuantiles: "
                "the ratio of sigmas is either too large or too small");

        if (quantileR >= 1.0/maxRatioNoCutoff_ && quantileR <= maxRatioNoCutoff_)
            // All is good, the support of the density does not have a sharp boundary
            return std::unique_ptr<RailwayGaussian>(new RailwayGaussian(
                median, sigmaPlus, sigmaMinus));
        else
        {
            const QuantileSigmaRatioFromR<RailwayGaussian> sigrat;
            const double target = quantileR > 1.0 ? quantileR : 1.0/quantileR;
            double inpR;
            const bool status = findRootUsingBisections(
                sigrat, target, maxRatioNoCutoff_, maxInputR,
                2.0*std::numeric_limits<double>::epsilon(), &inpR);
            assert(status);
            double sigmaP0 = 2.0*inpR/(1.0 + inpR);
            double sigmaM0 = 2.0/(1.0 + inpR);
            if (quantileR < 1.0)
                std::swap(sigmaP0, sigmaM0);
            const std::pair<double,double>& h =
                RailwayGaussian::transitionRegionChoice(sigmaP0, sigmaM0);
            const RailwayGaussian d0(0.0, sigmaP0, sigmaM0, h.first, h.second);
            const double med = d0.quantile(0.5);
            const double qplus0 = d0.quantile(GCDF84);
            const double qminus0 = d0.quantile(GCDF16);
            const double scale = (sigmaPlus + sigmaMinus)/(qplus0 - qminus0);
            assert(scale > 0.0);
            const double newMed = median - med*scale;
            // Transition region sizes do not change if both
            // sigmas are scaled together
            return std::unique_ptr<RailwayGaussian>(new RailwayGaussian(
                newMed, sigmaP0*scale, sigmaM0*scale, h.first, h.second));
        }
    }

    std::pair<double,double> RailwayGaussian::transitionRegionChoice(
        const double sigmaPlus, const double sigmaMinus)
    {
        static const double minH = 0.1;
        static const double maxH = 10.0;

        const double alpha = (sigmaPlus - sigmaMinus)/2.0;
        const double secondDerivative = 2.0*alpha;
        double leftH = maxH, rightH = maxH;
        if (secondDerivative)
        {
            // The choice given here will reduce the "final"
            // derivative by factor of 2 (compared to the
            // derivative at the boundary) in the important
            // cases where DistortedGaussian would get an
            // extremum outside of the [-1, 1] region.
            // However, we also don't want the h to get
            // very small, so there is a limit from below.
            const double sigma = (sigmaPlus + sigmaMinus)/2.0;
            const double leftDerivative = sigma - secondDerivative;
            const double rightDerivative = sigma + secondDerivative;

            leftH = std::abs(leftDerivative/secondDerivative);
            // The std::clamp function is not yet in C++11
            if (leftH < minH) leftH = minH;
            else if (leftH > maxH) leftH = maxH;

            rightH = std::abs(rightDerivative/secondDerivative);
            if (rightH < minH) rightH = minH;
            else if (rightH > maxH) rightH = maxH;
        }

        return std::pair<double,double>(leftH, rightH);
    }

    RailwayGaussian::RailwayGaussian(const double i_location,
                                     const double i_sigmaPlus,
                                     const double i_sigmaMinus,
                                     const double i_hleft,
                                     const double i_hright)
        : Base(i_location, i_sigmaPlus, i_sigmaMinus,
               Transform(i_sigmaPlus, i_sigmaMinus, i_hleft, i_hright,
                         (std::abs(i_sigmaPlus) + std::abs(i_sigmaMinus))/2.0))
    {
    }

    RailwayGaussian::RailwayGaussian(const double i_location,
                                     const double i_sigmaPlus,
                                     const double i_sigmaMinus)
        : Base(i_location, i_sigmaPlus, i_sigmaMinus,
               Transform(i_sigmaPlus, i_sigmaMinus,
                         transitionRegionChoice(i_sigmaPlus, i_sigmaMinus).first,
                         transitionRegionChoice(i_sigmaPlus, i_sigmaMinus).second,
                         (std::abs(i_sigmaPlus) + std::abs(i_sigmaMinus))/2.0))
    {
    }

    RailwayGaussian::RailwayGaussian(const std::vector<double>& cumulants)
        : Base(cumulants.at(0), sqrt(cumulants.at(1)), sqrt(cumulants.at(1)),
               Transform(1.0, 1.0, 10.0, 10.0))
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            static const double smallestSkew = skewnessForSmallR(0.0);

            const double negskew = -std::abs(skew);
            if (negskew <= smallestSkew)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::RailwayGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << -smallestSkew << '.';
                throw std::invalid_argument(os.str());
            }

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double r;
            const bool status = findRootUsingBisections(
                DoubleFunctor1(skewnessForSmallR), negskew, 0.0, 1.0, tol, &r);
            if (!status) throw std::runtime_error(
                "In ase::RailwayGaussian constructor: root finding failed");
            double sp = 2.0*r/(1.0 + r);
            double sm = 2.0/(1.0 + r);
            if (skew > 0.0)
                std::swap(sp, sm);
            const std::pair<double,double>& h = transitionRegionChoice(sp, sm);
            const RailwayGaussian d(0.0, sp, sm, h.first, h.second);
            const long double mu = d.calculateMoment(0.0L, 1U);
            const double var = d.calculateMoment(mu, 2U);
            const double scale = stdev/sqrt(var);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - static_cast<double>(mu)*scale);
            sigmaPlus_ = sp*scale;
            sigmaMinus_ = sm*scale;
            tr_ = Transform(sp, sm, h.first, h.second);
            Base::initialize();
        }
    }

    double RailwayGaussian::skewnessForSmallR(const double r)
    {
        assert(r >= 0.0);
        assert(r <= 1.0);
        if (r == 1.0)
            return 0.0;

        const double sp = 2.0*r/(1.0 + r);
        const double sm = 2.0/(1.0 + r);
        const std::pair<double,double>& h = transitionRegionChoice(sp, sm);
        const RailwayGaussian d(0.0, sp, sm, h.first, h.second);
        const long double mu = d.calculateMoment(0.0L, 1U);
        const long double var = d.calculateMoment(mu, 2U);
        assert(var > 0.0L);
        const long double mu3 = d.calculateMoment(mu, 3U);
        return mu3/powl(var, 1.5L);
    }

    long double RailwayGaussian::calculateMoment(const long double mu,
                                                 const unsigned power) const
    {
        static const unsigned maxdegAll = 12U;

        long double coeffs[maxdegAll+1U];
        const HermiteProbOrthoPoly he;
        long double sum = 0.0L;

        {
            // Leftmost zone integral
            const RailwayZoneFunctor<long double> fcn(
                tr_, -1.0 - 2.0*tr_.hleft(), mu, power);
            const unsigned maxdeg = power;
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += he.weightedSeriesIntegral(coeffs, maxdeg, -1.0 - tr_.hleft());
        }

        {
            // Left transition zone integral
            const RailwayZoneFunctor<long double> fcn(
                tr_, -1.0 - 0.5*tr_.hleft(), mu, power);
            const unsigned maxdeg = power*3;
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, -1.0)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, -1.0 - tr_.hleft()));
        }

        {
            // Central zone integral
            const RailwayZoneFunctor<long double> fcn(
                tr_, 0.0, mu, power);
            const unsigned maxdeg = power*2;
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, 1.0)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, -1.0));
        }

        {
            // Right transition zone integral
            const RailwayZoneFunctor<long double> fcn(
                tr_, 1.0 + 0.5*tr_.hright(), mu, power);
            const unsigned maxdeg = power*3;
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, 1.0 + tr_.hright())
                    - he.weightedSeriesIntegral(coeffs, maxdeg, 1.0));
        }

        {
            // Rightmost zone integral
            static const long double effectiveInfinity = 2.0L*inverseGaussCdf(1.0);

            const RailwayZoneFunctor<long double> fcn(
                tr_, 1.0 + 2.0*tr_.hright(), mu, power);
            const unsigned maxdeg = power;
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, effectiveInfinity)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, 1.0 + tr_.hright()));
        }

        return sum;
    }

    /********************************************************************/

    std::unique_ptr<DoubleCubicGaussian>
    DoubleCubicGaussian::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        validateDeltas("ase::DoubleCubicGaussian::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);

        if (deltaPlus == deltaMinus)
        {
            const double sig = deltaPlus/sqrt(2.0*deltaLnL);
            return std::unique_ptr<DoubleCubicGaussian>(
                new DoubleCubicGaussian(mode, sig, sig));
        }

        const double maxInputRatio = 1.8352553;
        const DescentDeltaRatioFromR<DoubleCubicGaussian> delRatio(deltaLnL);
        const double maxDelRatio = delRatio(maxInputRatio);
        const double rbig = std::max(deltaPlus, deltaMinus)/std::min(deltaPlus, deltaMinus);
        if (rbig >= maxDelRatio)
        {
            std::ostringstream os;
            os.precision(10);
            os << "In ase::DoubleCubicGaussian::fromModeAndDeltas: "
               << "the ratio of descent deltas, " << rbig << ", is too large. "
               << "It must be less than " << maxDelRatio << '.';
            throw std::invalid_argument(os.str());
        }
        double inpR;
        const bool status = findRootUsingBisections(
            delRatio, rbig, 1.0, maxInputRatio,
            2.0*std::numeric_limits<double>::epsilon(), &inpR);
        assert(status);
        double sigmaP0 = 2.0*inpR/(1.0 + inpR);
        double sigmaM0 = 2.0/(1.0 + inpR);
        if (deltaPlus/deltaMinus < 1.0)
            std::swap(sigmaP0, sigmaM0);
        const DoubleCubicGaussian distro(0.0, sigmaP0, sigmaM0);
        const double delPlus = distro.descentDelta(true, deltaLnL);
        const double delMinus = distro.descentDelta(false, deltaLnL);
        const double scale = (deltaPlus + deltaMinus)/(delPlus + delMinus);
        assert(scale > 0.0);
        const double unscaledMode = distro.mode();
        const double newMed = mode - unscaledMode*scale;
        return std::unique_ptr<DoubleCubicGaussian>(new DoubleCubicGaussian(
            newMed, sigmaP0*scale, sigmaM0*scale));
    }

    std::unique_ptr<DoubleCubicGaussian>
    DoubleCubicGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        static const double maxQuantileR = 6.854844;
        static const double maxInputR = 7.62;

        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::DoubleCubicGaussian::fromQuantiles: "
            "sigma parameters must be positive");
        const double quantileR = sigmaPlus/sigmaMinus;
        if (quantileR > maxQuantileR || quantileR < 1.0/maxQuantileR)
            throw std::invalid_argument(
                "In ase::DoubleCubicGaussian::fromQuantiles: "
                "the ratio of sigmas is either too large or too small");

        const double maxRatioNoCutoff = 5.0;
        if (quantileR >= 1.0/maxRatioNoCutoff && quantileR <= maxRatioNoCutoff)
        {
            return std::unique_ptr<DoubleCubicGaussian>(
                new DoubleCubicGaussian(median, sigmaPlus, sigmaMinus));
        }
        else
        {
            const QuantileSigmaRatioFromR<DoubleCubicGaussian> sigrat;
            const double target = quantileR > 1.0 ? quantileR : 1.0/quantileR;
            double inpR;
            const bool status = findRootUsingBisections(
                sigrat, target, maxRatioNoCutoff, maxInputR,
                2.0*std::numeric_limits<double>::epsilon(), &inpR);
            assert(status);
            double sigmaP0 = 2.0*inpR/(1.0 + inpR);
            double sigmaM0 = 2.0/(1.0 + inpR);
            if (quantileR < 1.0)
                std::swap(sigmaP0, sigmaM0);
            const DoubleCubicGaussian d0(0.0, sigmaP0, sigmaM0);
            const double med = d0.quantile(0.5);
            const double qplus0 = d0.quantile(GCDF84);
            const double qminus0 = d0.quantile(GCDF16);
            const double scale = (sigmaPlus + sigmaMinus)/(qplus0 - qminus0);
            assert(scale > 0.0);
            const double newMed = median - med*scale;
            return std::unique_ptr<DoubleCubicGaussian>(
                new DoubleCubicGaussian(newMed, sigmaP0*scale, sigmaM0*scale));
        }
    }

    DoubleCubicGaussian::DoubleCubicGaussian(const double i_median,
                                             const double i_sigmaPlus,
                                             const double i_sigmaMinus)
        : Base(i_median, i_sigmaPlus, i_sigmaMinus,
               Transform(i_sigmaPlus, i_sigmaMinus, (std::abs(i_sigmaPlus) +
                                                     std::abs(i_sigmaMinus))/2.0))
    {
    }

    double DoubleCubicGaussian::skewnessForSmallR(const double r)
    {
        assert(r >= 0.0);
        assert(r <= 1.0);
        if (r == 1.0)
            return 0.0;

        const double sp = 2.0*r/(1.0 + r);
        const double sm = 2.0/(1.0 + r);
        const DoubleCubicGaussian d(0.0, sp, sm);
        const long double mu = d.calculateMoment(0.0L, 1U);
        const long double var = d.calculateMoment(mu, 2U);
        assert(var > 0.0L);
        const long double mu3 = d.calculateMoment(mu, 3U);
        return mu3/powl(var, 1.5L);
    }

    DoubleCubicGaussian::DoubleCubicGaussian(
        const std::vector<double>& cumulants)
        : Base(cumulants.at(0), sqrt(cumulants.at(1)), sqrt(cumulants.at(1)),
               Transform(1.0, 1.0))
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            static const double smallestSkew = skewnessForSmallR(0.0);

            const double negskew = -std::abs(skew);
            if (negskew <= smallestSkew)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::DoubleCubicGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << -smallestSkew << '.';
                throw std::invalid_argument(os.str());
            }

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double r;
            const bool status = findRootUsingBisections(
                DoubleFunctor1(skewnessForSmallR), negskew, 0.0, 1.0, tol, &r);
            if (!status) throw std::runtime_error(
                "In ase::DoubleCubicGaussian constructor: root finding failed");
            double sp = 2.0*r/(1.0 + r);
            double sm = 2.0/(1.0 + r);
            if (skew > 0.0)
                std::swap(sp, sm);
            const DoubleCubicGaussian d(0.0, sp, sm);
            const long double mu = d.calculateMoment(0.0L, 1U);
            const double var = d.calculateMoment(mu, 2U);
            const double scale = stdev/sqrt(var);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - static_cast<double>(mu)*scale);
            sigmaPlus_ = sp*scale;
            sigmaMinus_ = sm*scale;
            tr_ = Transform(sp, sm);
            Base::initialize();
        }
    }

    long double DoubleCubicGaussian::calculateMoment(const long double mu,
                                                     const unsigned power) const
    {
        static const unsigned maxdegAll = 12U;

        long double coeffs[maxdegAll+1U];
        const HermiteProbOrthoPoly he;
        long double sum = 0.0L;

        {
            // Leftmost zone integral
            const SDCZoneFunctor<long double> fcn(tr_, -2.0, mu, power);
            const unsigned maxdeg = power;
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += he.weightedSeriesIntegral(coeffs, maxdeg, -1.0);
        }

        {
            // Left cubic zone integral
            const SDCZoneFunctor<long double> fcn(tr_, -0.5, mu, power);
            const unsigned maxdeg = power*3;
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, 0.0)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, -1.0));
        }

        {
            // Right cubic zone integral
            const SDCZoneFunctor<long double> fcn(tr_, 0.5, mu, power);
            const unsigned maxdeg = power*3;
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, 1.0)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, 0.0));
        }

        {
            // Rightmost zone integral
            static const long double effectiveInfinity = 2.0L*inverseGaussCdf(1.0);

            const SDCZoneFunctor<long double> fcn(tr_, 2.0, mu, power);
            const unsigned maxdeg = power;
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, effectiveInfinity)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, 1.0));
        }

        return sum;
    }

    /********************************************************************/

    std::unique_ptr<SymmetricBetaGaussian>
    SymmetricBetaGaussian::fromModeAndDeltas(
        const double mode, const double deltaPlus, const double deltaMinus,
        const unsigned p, const double h, const double deltaLnL)
    {
        validateDeltas("ase::SymmetricBetaGaussian::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);

        if (deltaPlus == deltaMinus)
        {
            const double sig = deltaPlus/sqrt(2.0*deltaLnL);
            return std::unique_ptr<SymmetricBetaGaussian>(
                new SymmetricBetaGaussian(mode, sig, sig, p, h));
        }

        const std::pair<double,double>& limit = minDescentDeltaRatio(p, h, deltaLnL);
        const double target = std::min(deltaPlus, deltaMinus)/std::max(deltaPlus, deltaMinus);
        if (target < limit.second)
        {
            std::ostringstream os;
            os.precision(16);
            os << "In ase::SymmetricBetaGaussian::fromModeAndDeltas: "
               << "the ratio of sigmas is either too large or too small. "
               << "For p = " << p << ", h = " << h
               << " and delta ln(L) = " << deltaLnL
               << " it must not exceed " << 1.0/limit.second << '.';
            throw std::invalid_argument(os.str());
        }

        const DescentDeltaRatioFromRSBG delRatio(deltaLnL, p, h);
        double inpR;
        const bool status = findRootUsingBisections(
            delRatio, target, limit.first, 1.0,
            2.0*std::numeric_limits<double>::epsilon(), &inpR);
        assert(status);
        double sigmaP0 = 2.0*inpR/(1.0 + inpR);
        double sigmaM0 = 2.0/(1.0 + inpR);
        if (deltaPlus/deltaMinus > 1.0)
            std::swap(sigmaP0, sigmaM0);
        const SymmetricBetaGaussian distro(0.0, sigmaP0, sigmaM0, p, h);
        const double delPlus = distro.descentDelta(true, deltaLnL);
        const double delMinus = distro.descentDelta(false, deltaLnL);
        const double scale = (deltaPlus + deltaMinus)/(delPlus + delMinus);
        assert(scale > 0.0);
        const double unscaledMode = distro.mode();
        const double newMed = mode - unscaledMode*scale;
        return std::unique_ptr<SymmetricBetaGaussian>(new SymmetricBetaGaussian(
            newMed, sigmaP0*scale, sigmaM0*scale, p, h));
    }

    std::pair<double,double> SymmetricBetaGaussian::minDescentDeltaRatio(
        const unsigned p, const double h, const double deltaLnL)
    {
        const DescentDeltaRatioFromRSBG ddrat(deltaLnL, p, h);

        // Try to find a good bracket for the minimum
        const double rMinNoExt = SymbetaDoubleIntegral<double>::minRNoExtremum(p, h);
        assert(rMinNoExt < 1.0);
        const double factor = M_SQRT2;
        double step = 1.0 - rMinNoExt;
        double ratio = 1.0;
        bool bracketed = false;
        for (unsigned i=0; i<200U && !bracketed; ++i)
        {
            step /= factor;
            const double rtry = rMinNoExt + step;
            const double f = ddrat(rtry);
            if (f > ratio)
                bracketed = true;
            else
                ratio = f;
        }
        if (!bracketed) throw std::runtime_error(
            "In ase::SymmetricBetaGaussian::minDescentDeltaRatio: "
            "failed to bracket the minimum");

        // Determine the minimum
        const double rLeft = rMinNoExt + step;
        const double rCur = rMinNoExt + step*factor;
        const double rRight = rMinNoExt + step*factor*factor;
        double argmin, fmin;
        const double tol = sqrt(std::numeric_limits<double>::epsilon());
        const bool status = findMinimumGoldenSection(
            ddrat, rLeft, rCur, rRight, tol, &argmin, &fmin);
        assert(status);
        return std::make_pair(argmin, fmin);
    }

    std::pair<double,double> SymmetricBetaGaussian::minQuantileRatio(
        const unsigned p, const double h)
    {
        const QuantileSigmaRatioFromRSBG sigrat(p, h);

        // Check the derivative at 0. If it is non-negative,
        // the minimum is likely at 0.
        const volatile double step = 1.0e-5;
        const double sigrat0 = sigrat(0.0);
        const double deriv0 = (sigrat(step) - sigrat0)/step;
        if (deriv0 >= 0.0)
            return std::make_pair(0.0, sigrat0);

        // Determine the minimum
        double argmin, fmin;
        const double tol = sqrt(std::numeric_limits<double>::epsilon());
        const bool status = findMinimumGoldenSection(
            sigrat, 0.0, step, 1.0, tol, &argmin, &fmin);
        assert(status);
        return std::make_pair(argmin, fmin);
    }

    std::unique_ptr<SymmetricBetaGaussian>
    SymmetricBetaGaussian::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus,
        const unsigned p, const double h)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::SymmetricBetaGaussian::fromQuantiles: "
            "sigma parameters must be positive");

        std::unique_ptr<SymmetricBetaGaussian> ptry(
            new SymmetricBetaGaussian(median, sigmaPlus, sigmaMinus, p, h));
        if (ptry->isUnimodal())
            return ptry;
        else
        {
            const double quantileR = sigmaPlus/sigmaMinus;
            const double target = quantileR > 1.0 ? 1.0/quantileR : quantileR;
            const std::pair<double,double>& limit = minQuantileRatio(p, h);
            if (target < limit.second)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::SymmetricBetaGaussian::fromQuantiles: "
                   << "the ratio of sigmas is either too large or too small. "
                   << "For p = " << p << " and h = " << h << " it must not exceed "
                   << 1.0/limit.second << '.';
                throw std::invalid_argument(os.str());
            }

            double inpR = limit.first;
            if (target > limit.second)
            {
                const QuantileSigmaRatioFromRSBG sigrat(p, h);
                const bool status = findRootUsingBisections(
                    sigrat, target, inpR, 1.0,
                    2.0*std::numeric_limits<double>::epsilon(), &inpR);
                assert(status);
            }

            double sigmaP0 = 2.0*inpR/(1.0 + inpR);
            double sigmaM0 = 2.0/(1.0 + inpR);
            if (quantileR > 1.0)
                std::swap(sigmaP0, sigmaM0);

            const SymmetricBetaGaussian d0(0.0, sigmaP0, sigmaM0, p, h);
            const double med = d0.quantile(0.5);
            const double qplus0 = d0.quantile(GCDF84);
            const double qminus0 = d0.quantile(GCDF16);
            const double scale = (sigmaPlus + sigmaMinus)/(qplus0 - qminus0);
            assert(scale > 0.0);
            const double newMed = median - med*scale;
            return std::unique_ptr<SymmetricBetaGaussian>(new SymmetricBetaGaussian(
                newMed, sigmaP0*scale, sigmaM0*scale, p, h));
        }
    }

    SymmetricBetaGaussian::SymmetricBetaGaussian(const double i_median,
                                                 const double i_sigmaPlus,
                                                 const double i_sigmaMinus,
                                                 const unsigned i_p,
                                                 const double i_h)
        : Base(i_median, i_sigmaPlus, i_sigmaMinus,
               Transform::fromSigmas(i_p, i_h, i_sigmaPlus, i_sigmaMinus, true))
    {
        if (!i_p) throw std::invalid_argument(
            "In ase::SymmetricBetaGaussian constructor: "
            "parameter p is expected to be positive");
    }

    double SymmetricBetaGaussian::skewnessForSmallR(
        const double r, const unsigned p, const double h)
    {
        assert(r >= 0.0);
        assert(r <= 1.0);
        if (r == 1.0)
            return 0.0;

        const double sp = 2.0*r/(1.0 + r);
        const double sm = 2.0/(1.0 + r);
        const SymmetricBetaGaussian d(0.0, sp, sm, p, h);
        const long double mu = d.calculateMoment(0.0L, 1U);
        const long double var = d.calculateMoment(mu, 2U);
        assert(var > 0.0L);
        const long double mu3 = d.calculateMoment(mu, 3U);
        return mu3/powl(var, 1.5L);
    }

    SymmetricBetaGaussian::SymmetricBetaGaussian(
        const std::vector<double>& cumulants,
        const unsigned p, const double h)
        : Base(cumulants.at(0), sqrt(cumulants.at(1)), sqrt(cumulants.at(1)), Transform())
    {
        if (!p) throw std::invalid_argument(
            "In ase::SymmetricBetaGaussian constructor: "
            "parameter p is expected to be positive");

        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            const double smallestSkew = skewnessForSmallR(0.0, p, h);
            const double negskew = -std::abs(skew);
            if (negskew <= smallestSkew)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::SymmetricBetaGaussian constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << -smallestSkew << '.';
                throw std::invalid_argument(os.str());
            }

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double r;
            const bool status = findRootUsingBisections(
                SkewFcn(p, h), negskew, 0.0, 1.0, tol, &r);
            if (!status) throw std::runtime_error(
                "In ase::SymmetricBetaGaussian constructor: root finding failed");
            double sp = 2.0*r/(1.0 + r);
            double sm = 2.0/(1.0 + r);
            if (skew > 0.0)
                std::swap(sp, sm);
            const SymmetricBetaGaussian d(0.0, sp, sm, p, h);
            const long double mu = d.calculateMoment(0.0L, 1U);
            const double var = d.calculateMoment(mu, 2U);
            const double scale = stdev/sqrt(var);
            AbsLocationScaleFamily::setScale(scale);
            AbsLocationScaleFamily::setLocation(cumulants[0] - static_cast<double>(mu)*scale);
            sigmaPlus_ = sp*scale;
            sigmaMinus_ = sm*scale;
            tr_ = Transform::fromSigmas(p, h, sp, sm);
            Base::initialize();
        }
    }

    long double SymmetricBetaGaussian::calculateMoment(const long double mu,
                                                       const unsigned power) const
    {
        static const unsigned maxdegAll = 168U;

        long double coeffs[maxdegAll+1U];
        const HermiteProbOrthoPoly he;
        const long double h = tr_.h();
        long double sum = 0.0L;

        {
            // Left zone integral
            const SDIZoneFunctor<long double> fcn(tr_, -2.0*h, mu, power);
            const unsigned maxdeg = power;
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += he.weightedSeriesIntegral(coeffs, maxdeg, -h);
        }

        {
            // Central zone integral
            const SDIZoneFunctor<long double> fcn(tr_, 0.0, mu, power);
            const unsigned maxdeg = power*(tr_.p()*2U + 2U);
            assert(maxdeg <= maxdegAll);
            he.calculateCoeffs(fcn, coeffs, maxdeg);
            sum += (he.weightedSeriesIntegral(coeffs, maxdeg, h)
                    - he.weightedSeriesIntegral(coeffs, maxdeg, -h));
        }

        {
            // Right zone integral
            static const long double effectiveInfinity = 2.0L*inverseGaussCdf(1.0);

            if (h < effectiveInfinity)
            {
                const SDIZoneFunctor<long double> fcn(tr_, 2.0*h, mu, power);
                const unsigned maxdeg = power;
                he.calculateCoeffs(fcn, coeffs, maxdeg);
                sum += (he.weightedSeriesIntegral(coeffs, maxdeg, effectiveInfinity)
                        - he.weightedSeriesIntegral(coeffs, maxdeg, h));
            }
        }

        return sum;
    }

    /********************************************************************/

    EmpiricalDistribution::EmpiricalDistribution(
        const std::vector<double>& i_sample, const bool sorted)
        : sortedSample_(i_sample)
    {
        if (i_sample.empty()) throw std::invalid_argument(
            "In ase::EmpiricalDistribution constructor: "
            "input sample is empty");            
        if (!sorted)
            std::sort(sortedSample_.begin(), sortedSample_.end());

        long double cums[5U];
        arrayCumulants(&sortedSample_[0], sortedSample_.size(), 4U, cums);
        for (unsigned i=0; i<5U; ++i)
            cumulants_[i] = cums[i];
    }

    double EmpiricalDistribution::density(double /* x */) const
    {
        throw std::runtime_error("In ase::EmpiricalDistribution::density: "
                                 "density estimation is not supported");
        return 0.0;
    }

    double EmpiricalDistribution::densityDerivative(double /* x */) const
    {
        throw std::runtime_error("In ase::EmpiricalDistribution::densityDerivative: "
                                 "density estimation is not supported");
        return 0.0;
    }

    double EmpiricalDistribution::mode() const
    {
        throw std::runtime_error("In ase::EmpiricalDistribution::mode: "
                                 "density estimation is not supported");
        return 0.0;
    }

    double EmpiricalDistribution::descentDelta(
        bool /* isToTheRight */, double /* deltaLnL */) const
    {
        throw std::runtime_error("In ase::EmpiricalDistribution::descentDelta: "
                                 "density estimation is not supported");
        return 0.0;
    }

    double EmpiricalDistribution::cdf(const double x) const
    {
        if (x < sortedSample_[0])
            return 0.0;
        if (x >= sortedSample_.back())
            return 1.0;
        const unsigned long iabove =
            std::upper_bound(sortedSample_.begin(), sortedSample_.end(), x) -
            sortedSample_.begin();
        return static_cast<double>(iabove)/sortedSample_.size();
    }

    double EmpiricalDistribution::exceedance(const double x) const
    {
        return 1.0 - cdf(x);
    }

    double EmpiricalDistribution::quantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::EmpiricalDistribution::quantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return sortedSample_[0];
        if (r1 == 1.0)
            return sortedSample_.back();
        const unsigned long sz = sortedSample_.size();
        if (sz == 1UL)
            return sortedSample_[0];
        const unsigned long idx = r1*sz;
        assert(idx < sz);
        return sortedSample_[idx];
    }

    double EmpiricalDistribution::invExceedance(const double r1) const
    {
        return quantile(1.0 - r1);
    }

    double EmpiricalDistribution::random(AbsRNG& gen) const
    {
        const unsigned long sz = sortedSample_.size();
        unsigned long idx = sz;
        while (idx >= sz)
            idx = gen()*sz;
        return sortedSample_[idx];
    }

    double EmpiricalDistribution::cumulant(const unsigned n) const
    {
        if (n > 4U)
            throw std::invalid_argument(
                "In ase::EmpiricalDistribution::cumulant: "
                "only four leading cumulants are implemented");
        return cumulants_[n];
    }

    /********************************************************************/
    
    std::unique_ptr<JohnsonSystem> JohnsonSystem::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        static const double maxQuantileR = 2.0637;

        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::JohnsonSystem::fromQuantiles: "
            "sigma parameters must be positive");

        if (sigmaPlus == sigmaMinus)
            return std::unique_ptr<JohnsonSystem>(new JohnsonSystem(
                median, sigmaPlus, 0.0, 3.0));

        const double quantileR = sigmaPlus/sigmaMinus;
        if (quantileR > maxQuantileR || quantileR < 1.0/maxQuantileR)
            throw std::invalid_argument(
                "In ase::JohnsonSystem::fromQuantiles: "
                "the ratio of sigmas is too large");

        return buildFromMedianAndSigmas<JohnsonSystem>(
            median, sigmaPlus, sigmaMinus, maxAutoSkew_);
    }

    std::unique_ptr<JohnsonSystem> JohnsonSystem::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        // The ratio of deltas (calculated at deltaLnL = 0.5)
        // starts decreasing above the following skewness
        const double maxInputSkew = 2.2967561767492;

        validateDeltas("ase::JohnsonSystem::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);
        validateDeltasRatio<JohnsonSystem>(
            "ase::JohnsonSystem::fromModeAndDeltas",
            deltaPlus, deltaMinus, deltaLnL, maxInputSkew);

        return buildFromModeAndDeltas<JohnsonSystem>(
            mode, deltaPlus, deltaMinus, deltaLnL, maxInputSkew);
    }

    /********************************************************************/

    std::unique_ptr<EdgeworthExpansion3> EdgeworthExpansion3::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::EdgeworthExpansion3::fromQuantiles: "
            "sigma parameters must be positive");

        const double skewMax = largestSkewAllowed(classSafeSigmaRange())*
                               (1.0 - 2.0*DBL_EPSILON);
        validateQSigmaRatio<EdgeworthExpansion3>(
            "ase::EdgeworthExpansion3::fromQuantiles",
            sigmaPlus, sigmaMinus, skewMax);

        return buildFromMedianAndSigmas<EdgeworthExpansion3>(
            median, sigmaPlus, sigmaMinus, skewMax);
    }

    /********************************************************************/

    double FechnerDistribution::unscaledCumulant(const unsigned n) const
    {
        if (n)
        {
            assert(sPstd_ >= 0.0);
            assert(sMstd_ >= 0.0);

            if (sPstd_ == sMstd_)
                return n == 2U ? 1.0 : 0.0;
            else
            {
                const double mean = (sPstd_ - sMstd_)*SQR2OVERPI;
                double cum = 0.0, ip = 0.0, in = 0.0;
                switch (n)
                {
                case 1U:
                    cum = mean;
                    break;
                case 2U:
                {
                    if (sPstd_)
                        ip = halfGaussInegral2(mean, sPstd_, true);
                    if (sMstd_)
                        in = halfGaussInegral2(mean, sMstd_, false);
                    cum = sPstd_*ip + sMstd_*in;
                }
                break;
                case 3U:
                {
                    if (sPstd_)
                        ip = halfGaussInegral3(mean, sPstd_, true);
                    if (sMstd_)
                        in = halfGaussInegral3(mean, sMstd_, false);
                    cum = sPstd_*ip + sMstd_*in;
                }
                break;
                case 4U:
                {
                    if (sPstd_)
                        ip = halfGaussInegral4(mean, sPstd_, true);
                    if (sMstd_)
                        in = halfGaussInegral4(mean, sMstd_, false);
                    const double k2 = unscaledCumulant(2U);
                    cum = sPstd_*ip + sMstd_*in - 3*k2*k2;
                }
                break;
                default:
                    throw std::invalid_argument(
                        "In ase::FechnerDistribution::unscaledCumulant: "
                        "only four leading cumulants are implemented");
                }
                return cum;
            }
        }
        else
            return 1.0;
    }

    std::unique_ptr<FechnerDistribution>
    FechnerDistribution::fromQuantiles(
        const double median, const double sigmaPlus, const double sigmaMinus)
    {
        if (sigmaPlus <= 0.0 || sigmaMinus <= 0.0) throw std::invalid_argument(
            "In ase::FechnerDistribution::fromQuantiles: "
            "sigma parameters must be positive");

        const double skewMax = largestSkew_*(1.0 - DBL_EPSILON);
        validateQSigmaRatio<FechnerDistribution>(
            "ase::FechnerDistribution::fromQuantiles",
            sigmaPlus, sigmaMinus, skewMax);

        const double aveSigma = (sigmaPlus + sigmaMinus)/2.0;
        if (std::abs(sigmaPlus - sigmaMinus)/aveSigma <= 2.0*DBL_EPSILON)
            return std::unique_ptr<FechnerDistribution>(
                new FechnerDistribution(median, aveSigma, aveSigma));

        return buildFromMedianAndSigmas<FechnerDistribution>(
            median, sigmaPlus, sigmaMinus, skewMax);
    }

    FechnerDistribution::FechnerDistribution(
        const double i_mu, const double i_sigmaPlus,
        const double i_sigmaMinus)
        : AbsLocationScaleFamily(i_mu, (std::abs(i_sigmaPlus) +
                                        std::abs(i_sigmaMinus))/2.0),
          sigmaPlus_(i_sigmaPlus),
          sigmaMinus_(i_sigmaMinus),
          g_(0.0, 1.0)
    {
        initialize();
    }

    void FechnerDistribution::initialize()
    {
        if (sigmaPlus_ < 0.0) throw std::invalid_argument(
            "In ase::FechnerDistribution::initialize: "
            "sigmaPlus parameter must be non-negative");
        if (sigmaMinus_ < 0.0) throw std::invalid_argument(
            "In ase::FechnerDistribution::initialize: "
            "sigmaMinus parameter must be non-negative");
        const double sc = (sigmaPlus_ + sigmaMinus_)/2.0;
        if (sc <= 0.0) std::invalid_argument(
            "In ase::FechnerDistribution::initialize: "
            "sum of sigmas must be positive");
        sPstd_ = sigmaPlus_/sc;
        sMstd_ = sigmaMinus_/sc;
        xmin_ = sMstd_*inverseGaussCdf(0.0);
        xmax_ = sPstd_*inverseGaussCdf(1.0);
    }

    FechnerDistribution::FechnerDistribution(const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(cumulants.at(0), sqrt(cumulants.at(1))),
          g_(0.0, 1.0)
    {
        const double k2 = cumulants[1];
        assert(k2 > 0.0);
        const double stdev = sqrt(k2);
        sigmaPlus_ = stdev;
        sigmaMinus_ = stdev;

        double skew = 0.0;
        if (cumulants.size() > 2U)
            skew = cumulants[2]/k2/stdev;
        if (skew)
        {
            const double absskew = std::abs(skew);
            if (absskew > largestSkew_)
            {
                std::ostringstream os;
                os.precision(16);
                os << "In ase::FechnerDistribution constructor: "
                   << "impossible set of cumulants. "
                   << "Normalized skewness magnitude must be below "
                   << largestSkew_ << '.';
                throw std::invalid_argument(os.str());
            }

            const double tol = 2.0*std::numeric_limits<double>::epsilon();
            double ratio;
            const bool status = findRootUsingBisections(
                DoubleFunctor1(skewFcn), absskew, 0.0, 1.0, tol, &ratio);
            if (!status) throw std::runtime_error(
                "In ase::FechnerDistribution constructor: root finding failed");

            const double sp = skew > 0.0 ? 1.0 : ratio;
            const double sm = skew > 0.0 ? ratio : 1.0;
            const FechnerDistribution fd(0.0, sp, sm);
            const double scale = stdev/sqrt(fd.cumulant(2));
            sigmaPlus_ = scale*sp;
            sigmaMinus_ = scale*sm;
            AbsLocationScaleFamily::setScale((sigmaPlus_ + sigmaMinus_)/2.0);
            AbsLocationScaleFamily::setLocation(cumulants[0] - scale*fd.cumulant(1));
        }
        initialize();
    }

    double FechnerDistribution::unscaledDensity(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        const double sig = x > 0.0 ? sPstd_ : sMstd_;
        const double del = x/sig;
        return exp(-del*del/2.0)/SQR2PI;
    }

    double FechnerDistribution::unscaledDensityDerivative(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        const double sig = x > 0.0 ? sPstd_ : sMstd_;
        const double del = x/sig;
        return -del*exp(-del*del/2.0)/SQR2PI/sig;
    }

    double FechnerDistribution::unscaledCdf(const double x) const
    {
        if (x < xmin_)
            return 0.0;
        else if (x > xmax_)
            return 1.0;
        else
        {
            if (x <= 0.0)
                return sMstd_*g_.cdf(x/sMstd_);
            else
                return 1.0 - sPstd_*g_.exceedance(x/sPstd_);
        }
    }

    double FechnerDistribution::unscaledExceedance(const double x) const
    {
        if (x < xmin_)
            return 1.0;
        else if (x > xmax_)
            return 0.0;
        else
        {
            if (x <= 0.0)
                return 1.0 - sMstd_*g_.cdf(x/sMstd_);
            else
                return sPstd_*g_.exceedance(x/sPstd_);
        }
    }

    double FechnerDistribution::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::FechnerDistribution::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return xmin_;
        if (r1 == 1.0)
            return xmax_;
        const double cdf0 = sMstd_/2.0;
        if (r1 < cdf0)
            return sMstd_*g_.quantile(r1/sMstd_);
        else if (r1 > cdf0)
            return sPstd_*g_.quantile((r1 + 1.0 - sMstd_)/sPstd_);
        else
            return 0.0;
    }

    double FechnerDistribution::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::FechnerDistribution::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return xmin_;
        if (r1 == 0.0)
            return xmax_;
        const double exc0 = sPstd_/2.0;
        if (r1 > exc0)
            return sMstd_*g_.invExceedance((r1 - 1.0 + sMstd_)/sMstd_);
        else if (r1 < exc0)
            return sPstd_*g_.invExceedance(r1/sPstd_);
        else
            return 0.0;
    }

    double FechnerDistribution::unscaledDescentDelta(
        const bool isToTheRight, const double deltaLnL) const
    {
        assert(deltaLnL > 0.0);
        return (isToTheRight ? sPstd_ : sMstd_)*sqrt(2.0*deltaLnL);
    }

    std::unique_ptr<FechnerDistribution>
    FechnerDistribution::fromModeAndDeltas(
        const double mode, const double deltaPlus,
        const double deltaMinus, const double deltaLnL)
    {
        validateDeltas("ase::FechnerDistribution::fromModeAndDeltas",
                       deltaPlus, deltaMinus, deltaLnL);
        const double factor = sqrt(2.0*deltaLnL);
        const double sp = deltaPlus/factor;
        const double sm = deltaMinus/factor;
        return std::unique_ptr<FechnerDistribution>(
            new FechnerDistribution(mode, sp, sm));
    }

    double FechnerDistribution::skewFcn(const double r)
    {
        assert(r >= 0.0 && r <= 1.0);
        if (r == 0.0)
            return largestSkew_ ;
        else if (r == 1.0)
            return 0.0;
        else
        {
            const FechnerDistribution fd(0.0, 1.0, r);
            const double k2 = fd.unscaledCumulant(2);
            assert(k2 > 0.0);
            return fd.unscaledCumulant(3)/k2/sqrt(k2);
        }
    }

    /********************************************************************/

    UniformDistribution::UniformDistribution(
        const double i_location, const double i_scale)
        : AbsLocationScaleFamily(i_location, i_scale)
    {
    }

    UniformDistribution::UniformDistribution(
        const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(0.0, sqrt(12.0*cumulants.at(1)))
    {
        AbsLocationScaleFamily::setLocation(cumulants[0] - 0.5*scale());
    }

    double UniformDistribution::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        switch (n)
        {
        case 0U:
            cum = 1.0;
            break;
        case 1U:
            cum = 0.5;
            break;
        case 2U:
            cum = 1.0/12.0;
            break;
        case 3U:
            cum = 0.0;
            break;
        case 4U:
            cum = -1.0/120.0;
            break;
        default:
            throw std::invalid_argument(
                "In ase::UniformDistribution::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    double UniformDistribution::unscaledDensity(const double x) const
    {
        if (x >= 0.0 && x < 1.0)
            return 1.0;
        else
            return 0.0;
    }

    double UniformDistribution::unscaledCdf(const double x) const
    {
        if (x <= 0.0)
            return 0.0;
        else if (x >= 1.0)
            return 1.0;
        else
            return x;
    }

    double UniformDistribution::unscaledExceedance(const double x) const
    {
        if (x <= 0.0)
            return 1.0;
        else if (x >= 1.0)
            return 0.0;
        else
            return 1.0 - x;
    }

    double UniformDistribution::unscaledMode() const
    {
        throw std::runtime_error("In ase::UniformDistribution::unscaledMode: "
                                 "the mode is undefined");
        return 0.0;
    }

    double UniformDistribution::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::UniformDistribution::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        return r1;
    }

    double UniformDistribution::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::UniformDistribution::unscaledInvExceedance: "
            "cdf argument outside of [0, 1] interval");
        return 1.0 - r1;
    }

    /********************************************************************/

    ExponentialDistribution::ExponentialDistribution(
        const double i_location, const double i_scale)
        : AbsLocationScaleFamily(i_location, i_scale)
    {
    }

    ExponentialDistribution::ExponentialDistribution(
        const std::vector<double>& cumulants)
        : AbsLocationScaleFamily(0.0, sqrt(cumulants.at(1)))
    {
        AbsLocationScaleFamily::setLocation(cumulants[0] - scale());
    }

    double ExponentialDistribution::unscaledCumulant(const unsigned n) const
    {
        double cum = 0.0;
        switch (n)
        {
        case 0U:
            cum = 1.0;
            break;
        case 1U:
            cum = 1.0;
            break;
        case 2U:
            cum = 1.0;
            break;
        case 3U:
            cum = 2.0;
            break;
        case 4U:
            cum = 6.0;
            break;
        default:
            throw std::invalid_argument(
                "In ase::ExponentialDistribution::unscaledCumulant: "
                "only four leading cumulants are implemented");
        }
        return cum;
    }

    double ExponentialDistribution::unscaledDensity(const double x) const
    {
        if (x >= 0.0)
        {
            const double eval = exp(-x);
            return eval < DBL_MIN ? 0.0 : eval;
        }
        else
            return 0.0;
    }

    double ExponentialDistribution::unscaledDensityDerivative(const double x) const
    {
        if (x >= 0.0)
        {
            const double eval = exp(-x);
            return eval < DBL_MIN ? 0.0 : -eval;
        }
        else
            return 0.0;
    }

    double ExponentialDistribution::unscaledCdf(const double x) const
    {
        if (x >= 0.0)
            return 1.0 - exp(-x);
        else
            return 0.0;
    }

    double ExponentialDistribution::unscaledExceedance(const double x) const
    {
        if (x >= 0.0)
        {
            const double eval = exp(-x);
            return eval < DBL_MIN ? 0.0 : eval;
        }
        else
            return 1.0;
    }

    double ExponentialDistribution::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::ExponentialDistribution::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return -log(DBL_MIN);
        else
            return -log(1.0 - r1);
    }

    double ExponentialDistribution::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::ExponentialDistribution::unscaledInvExceedance: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return -log(DBL_MIN);
        else
            return -log(r1);
    }
}
