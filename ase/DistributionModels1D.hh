#ifndef ASE_DISTRIBUTIONMODELS1D_HH_
#define ASE_DISTRIBUTIONMODELS1D_HH_

#include <utility>
#include <memory>

#include "ase/OPATGaussian.hh"
#include "ase/ParabolicRailwayCurve.hh"
#include "ase/SmoothDoubleCubic.hh"

namespace ase {
    /** Barlow's dimidated Gaussian, see arXiv:physics/0306138v1 */
    class DimidiatedGaussian : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = true;

        /**
        // Parameters sigmaPlus and sigmaMinus can be negative.
        // If the sigma is negative, the direction of the corresponding
        // Gaussian piece will be reversed and, of course, the "median"
        // parameter might no longer be the real median.
        */
        DimidiatedGaussian(double median, double sigmaPlus, double sigmaMinus);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than 2/sqrt(2*M_PI - 5).
        */
        explicit DimidiatedGaussian(const std::vector<double>& cumulants);

        inline virtual DimidiatedGaussian* clone() const override
            {return new DimidiatedGaussian(*this);}

        inline virtual ~DimidiatedGaussian() override {}

        inline double sigmaPlus() const {return sigmaPlus_;}
        inline double sigmaMinus() const {return sigmaMinus_;}
        inline double asymmetry() const
            {return (sigmaPlus_ - sigmaMinus_)/(sigmaPlus_ + sigmaMinus_);}

        inline virtual bool isDensityContinuous() const override
            {return sigmaPlus_ == sigmaMinus_;}

        virtual void setScale(double s) override;

        inline virtual std::string classname() const override
            {return "DimidiatedGaussian";}

        static std::unique_ptr<DimidiatedGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

    private:
        // The following functions will calculate unscaled skewness
        // as a function of normalized sigmaPlus (between 0 and 2)
        static double skewOneTail(double sp);
        static double skewTwoTails(double sp);

        void initialize(double sigmaPlus, double sigmaMinus);

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;
        inline virtual double unscaledMode() const override
            {return 0.0;}

        // Just remember these from the constructor
        double sigmaPlus_;
        double sigmaMinus_;

        // Actual parameters of the unscaled distribution
        double sPstd_;      // Both sPstd_ and sMstd_ are positive,
        double sMstd_;      // and their sum equals 2.0.
        double xmin_;
        double xmax_;

#ifdef SWIG
    public:
        inline static DimidiatedGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }
#endif
    };

    /** Barlow's distorted Gaussian, see arXiv:physics/0306138v1 */
    class DistortedGaussian : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = true;

        /** Parameters sigmaPlus and sigmaMinus can be negative */
        DistortedGaussian(double location, double sigmaPlus, double sigmaMinus);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than sqrt(8).
        */
        explicit DistortedGaussian(const std::vector<double>& cumulants);

        inline virtual DistortedGaussian* clone() const override
            {return new DistortedGaussian(*this);}

        inline virtual ~DistortedGaussian() override {}

        inline double sigmaPlus() const {return sigmaPlus_;}
        inline double sigmaMinus() const {return sigmaMinus_;}

        inline virtual bool isUnimodal() const override
            {return sigmaPlus_ == sigmaMinus_;}

        virtual void setScale(double s) override;

        inline virtual std::string classname() const override
            {return "DistortedGaussian";}

        static std::unique_ptr<DistortedGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

    private:
        // The following function will calculate unscaled skewness
        // as a function of sigma/alpha
        static double skewAlphaOneTail(double sigma);
        static double skewAlphaTwoTails(double alpha);

        void initialize(double sigmaPlus, double sigmaMinus);
        void findAndValidateRoots(double x, double* u1, double* u2) const;
        void calculateUnscaledCumulants() const;

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;

        // Just remember these two from the constructor
        double sigmaPlus_;
        double sigmaMinus_;

        double sPstd_;
        double sMstd_;
        double alpha_;
        double sigma_;
        double xmin_;
        double xmax_;
        Gaussian g_;

#ifdef SWIG
    public:
        inline static DistortedGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }
#endif
    };

    class RailwayGaussian : public OPATGaussian<ParabolicRailwayCurve<long double> >
    {
    public:
        typedef OPATGaussian<ParabolicRailwayCurve<long double> > Base;
        typedef typename Base::Transform Transform;

        /** Parameters sigmaPlus and sigmaMinus can be negative */
        RailwayGaussian(double location, double sigmaPlus, double sigmaMinus,
                        double hleft, double hright);

        /**
        // Constructor utilizing the default method of obtaining the sizes
        // of the transition regions, hleft, and hright
        */
        RailwayGaussian(double location, double sigmaPlus, double sigmaMinus);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than about 2.429336.
        */
        explicit RailwayGaussian(const std::vector<double>& cumulants);

        inline virtual RailwayGaussian* clone() const override
            {return new RailwayGaussian(*this);}

        inline virtual ~RailwayGaussian() override {}

        inline double hleft() const {return tr_.hleft();}
        inline double hright() const {return tr_.hright();}

        inline virtual std::string classname() const override
            {return "RailwayGaussian";}

        static std::unique_ptr<RailwayGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<RailwayGaussian> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

        // The default transition region choice. The first element of
        // the returned pair is hleft and the second element is hright.
        static std::pair<double,double> transitionRegionChoice(
            double sigmaPlus, double sigmaMinus);

    private:
        // The following constant must be in agreement with
        // "minH" and "maxH" definitions inside "transitionRegionChoice".
        // The transition region "h" that corresponds to 0 derivative
        // at the end of the transition region (this is the h
        // below which the distributions will have no cutoff)
        // is given by (3 - r)/(r - 1), where r = sigmaPlus/sigmaMinus.
        // Here, r is assumed to be larger than 1. "maxRatioNoCutoff_"
        // constant solves for r the equation (3 - r)/(r - 1) == minH.
        static const double maxRatioNoCutoff_;

        // Skewness for r = sigmaPlus/sigmaMinus values between 0 and 1
        // and the default transition region choice
        static double skewnessForSmallR(double r);

        virtual long double calculateMoment(
            long double mu, unsigned power) const override;

#ifdef SWIG
    public:
        inline static RailwayGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static RailwayGaussian* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    class DoubleCubicGaussian : public OPATGaussian<SmoothDoubleCubic<long double> >
    {
    public:
        typedef OPATGaussian<SmoothDoubleCubic<long double> > Base;
        typedef typename Base::Transform Transform;

        /**
        // Parameters sigmaPlus and sigmaMinus can be negative.
        // If the sigma is negative, the direction of the corresponding
        // Gaussian piece will be reversed and, of course, the "median"
        // parameter might no longer be the real median.
        */
        DoubleCubicGaussian(double median, double sigmaPlus, double sigmaMinus);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than 1.88785158.
        */
        explicit DoubleCubicGaussian(const std::vector<double>& cumulants);

        inline virtual DoubleCubicGaussian* clone() const override
            {return new DoubleCubicGaussian(*this);}

        inline virtual ~DoubleCubicGaussian() override {}

        inline virtual std::string classname() const override
            {return "DoubleCubicGaussian";}

        static std::unique_ptr<DoubleCubicGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<DoubleCubicGaussian> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

    private:
        // Skewness for r = sigmaPlus/sigmaMinus values between 0 and 1
        static double skewnessForSmallR(double r);

        virtual long double calculateMoment(
            long double mu, unsigned power) const override;

#ifdef SWIG
    public:
        inline static DoubleCubicGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static DoubleCubicGaussian* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /**
    // Skew-normal distribution,
    // see https://en.wikipedia.org/wiki/Skew_normal_distribution
    */
    class SkewNormal : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        SkewNormal(double location, double scale, double shapeParameter);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than
        // sqrt(2)*(4 - M_PI)/pow(M_PI - 2, 1.5).
        */
        explicit SkewNormal(const std::vector<double>& cumulants);

        inline virtual SkewNormal* clone() const override
            {return new SkewNormal(*this);}

        inline virtual ~SkewNormal() override {}

        inline double shapeParameter() const {return alpha_;}

        inline virtual std::string classname() const override
            {return "SkewNormal";}

        static std::unique_ptr<SkewNormal> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<SkewNormal> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

    private:
        // The following function calculates unscaled skewness
        // as a function of delta
        static double skewFcn(double delta);

        void initCorrections();
        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;

        static const double largestSkew_;
        
        // Notation follows Wikipedia
        double alpha_;
        double rngFactor_;
        double delta_;
        Gaussian g_;
        double xmin_;
        double xmax_;
        double cdfCorr_;
        double excCorr_;

#ifdef SWIG
    public:
        inline static SkewNormal* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static SkewNormal* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /**
    // A particular version of "simple Q-Normal distribution" by
    // Keelin and Powley which can be interpreted as a Gaussian
    // with variable width. See
    // 
    // https://en.wikipedia.org/wiki/Quantile-parameterized_distribution
    */
    class QVWGaussian : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        QVWGaussian(double location, double scale, double asymmetryParameter);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        //
        // Note that, for this distribution, the absolute value of
        // normalized skewness must be less than about ?
        */
        explicit QVWGaussian(const std::vector<double>& cumulants);

        inline virtual QVWGaussian* clone() const override
            {return new QVWGaussian(*this);}

        inline virtual ~QVWGaussian() override {}

        inline double asymmetryParameter() const {return a_;}

        // The following parameters are returned for
        // the density with zero mean and unit variance
        inline double locationParameter() const {return mu_;}
        inline double scaleParameter() const {return sigma_;}

        inline virtual std::string classname() const override
            {return "QVWGaussian";}

        static std::unique_ptr<QVWGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<QVWGaussian> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

    private:
        static void validateAsymmetry(double a);
        static long double moments(unsigned m, unsigned n);

        // The following function calculates unscaled skewness
        // as a function of asymmetry
        static double skewFcn(double a);

        void calcStandardParams(double a);

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;

        static const double largestSkew_;
        static const double largestAsymmetry_;
        
        // If Q(y) is the quantile function of the standard
        // normal, the quantile function of the QVWGaussian
        // will be mu + sigma*(1 + a*(y - 0.5))*Q(y). mu_ and
        // sigma_ in this implementation will be adjusted to get
        // the distribution with 0 mean and unit variance.
        double mu_;
        double sigma_;
        double a_;
        Gaussian g_;

#ifdef SWIG
    public:
        inline static QVWGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static QVWGaussian* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /** Gamma distribution */
    class GammaDistribution : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        // Chi-square distribution is GammaDistribution(0.0, 2.0, nDoF/2.0)
        GammaDistribution(double location, double scale, double shapeParameter);

        /**
        // The vector of cumulants must have the size of at least three.
        // If there are more than three cumulants, the excess cumulants
        // are ignored. Note that the third cumulant must be positive.
        */
        explicit GammaDistribution(const std::vector<double>& cumulants);

        inline virtual GammaDistribution* clone() const override
            {return new GammaDistribution(*this);}

        inline virtual ~GammaDistribution() override {}

        inline double shapeParameter() const {return alpha_;}

        inline virtual std::string classname() const override
            {return "GammaDistribution";}

        // The function "fromQuantiles" is not provided. The gamma
        // distribution does not tend to a Gaussian in the limit
        // sigmaPlus -> sigmaMinus (or skewness -> 0), and it is not
        // intended to be an asymmetric error model.

    private:
        static double unscaledRnd(double alpha, AbsRNG& gen);

        void initialize();

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        inline virtual double unscaledRandom(AbsRNG& gen) const override
            {return unscaledRnd(alpha_, gen);}

        double alpha_;
        double norm_;
        double uplim_;
    };

    /**
    // Lognormal distribution parameterized by its mean, standard
    // deviation, and normalised skewness
    */
    class LogNormal : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        LogNormal(double mean, double stdev, double skewness);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // If only two cumulants are provided, the third cumulant is
        // assumed to be zero (the resulting distribution is a Gaussian).
        // If there are more than three cumulants, the excess cumulants
        // are ignored.
        */
        explicit LogNormal(const std::vector<double>& cumulants);

        inline virtual LogNormal* clone() const override
            {return new LogNormal(*this);}

        inline virtual ~LogNormal() override {}

        inline double skewness() const {return skew_;}
        double kurtosis() const;
        inline bool isGaussian() const {return !skew_;}
        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        inline virtual std::string classname() const override
            {return "LogNormal";}

        static std::unique_ptr<LogNormal> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<LogNormal> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

    private:
        void initialize();

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;
        virtual double unscaledMode() const override;
        double unscaledEntropy() const;

        double skew_;
        double w_;
        double logw_;
        double s_;
        double xi_;
        double emgamovd_;

#ifdef SWIG
    public:
        inline static LogNormal* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static LogNormal* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /** Johnson S_u (unbounded) curve */
    class JohnsonSu : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        /**
        // For this distribution, not all combinations of skewness
        // and kurtosis are possible. Check the value of "isValid()"
        // function after the object is created in order to make sure
        // that the input parameters were acceptable.
        //
        // If the kurtosis is left at its default value of 0 (which
        // is impossible), it will be picked instead using the maximum
        // entropy principle. In this case the absolute value of
        // skewness parameter must not exceed about 730 (this is
        // a limitation of the code implementation, not something
        // imposed by the theory).
        */
        JohnsonSu(double location, double scale,
                  double skewness, double kurtosis = 0.0);

        /**
        // The vector of cumulants must have the size of at least three.
        // The fourth cumulant, if present, is the unnormalised excess
        // kurtosis. If there are more than four cumulants, the excess
        // cumulants are ignored. If three cumulants are provided, the
        // fourth cumulant is picked automatically using the maximum
        // entropy principle.
        */
        explicit JohnsonSu(const std::vector<double>& cumulants);

        inline virtual JohnsonSu* clone() const override
            {return new JohnsonSu(*this);}

        inline virtual ~JohnsonSu() override {}

        inline double skewness() const {return skew_;}
        inline double kurtosis() const {return kurt_;}
        inline bool isValid() const {return isValid_;}

        // Note that the parameters below are returned for
        // the density with zero mean and unit variance
        inline double getDelta() const {return delta_;}
        inline double getLambda() const {return lambda_;}
        inline double getGamma() const {return gamma_;}
        inline double getXi() const {return xi_;}

        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        inline virtual std::string classname() const override
            {return "JohnsonSu";}

        // The function "fromQuantiles" is not provided, as S_u
        // can not be constructed for the important case
        // sigmaPlus == sigmaMinus. Use JohnsonSystem class instead.

    private:
        void initialize();

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;
        double unscaledEntropy() const;

        double skew_;
        double kurt_;

        double delta_;
        double lambda_;
        double gamma_;
        double xi_;

        mutable double entropy_;
        mutable bool entropyCalculated_;

        bool isValid_;
    };

    /** Johnson S_b (bounded) curve */
    class JohnsonSb : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        /**
        // For this distribution, not all combinations of skewness
        // and kurtosis are possible. Check the value of "isValid()"
        // function after the object is created in order to make sure
        // that the input parameters were acceptable.
        */
        JohnsonSb(double location, double scale,
                  double skewness, double kurtosis);

        /**
        // The vector of cumulants must have the size of at least three.
        // The fourth cumulant, if present, is the unnormalised excess
        // kurtosis. If there are more than four cumulants, the excess
        // cumulants are ignored. If three cumulants are provided, the
        // fourth cumulant is assumed to be zero (that is, the kurtosis
        // is set to 3).
        */
        explicit JohnsonSb(const std::vector<double>& cumulants);

        inline virtual JohnsonSb* clone() const override
            {return new JohnsonSb(*this);}

        inline virtual ~JohnsonSb() override {}

        inline double skewness() const {return skew_;}
        inline double kurtosis() const {return kurt_;}
        inline bool isValid() const {return isValid_;}

        virtual bool isUnimodal() const override;
        
        // Note that the parameters below are returned for
        // the density with zero mean and unit variance
        inline double getDelta() const {return delta_;}
        inline double getLambda() const {return lambda_;}
        inline double getGamma() const {return gamma_;}
        inline double getXi() const {return xi_;}

        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        inline virtual std::string classname() const override
            {return "JohnsonSb";}

        // The function "fromQuantiles" is not provided, as S_b
        // can not be constructed for the important case
        // sigmaPlus == sigmaMinus. Use JohnsonSystem class instead.

        static bool fitParameters(double skewness, double kurtosis,
                                  double *gamma, double *delta,
                                  double *lambda, double *xi);

    private:
        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledRandom(AbsRNG& gen) const override;
        double unscaledEntropy() const;

        double skew_;
        double kurt_;

        double delta_;
        double lambda_;
        double gamma_;
        double xi_;

        mutable double entropy_;
        mutable bool entropyCalculated_;

        bool isValid_;
    };

    /** This class selects an appropriate Johnson curve automatically */
    class JohnsonSystem : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        enum CurveType {
            GAUSSIAN = 0,
            LOGNORMAL,
            SU,
            SB,
            INVALID
        };

        // This constructor will throw std::invalid_argument if the
        // combination of skewness and kurtosis arguments is impossible
        JohnsonSystem(double location, double scale,
                      double skewness, double kurtosis);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. The third element, if present,
        // is the unnormalised skewness (the third central moment).
        // The fourth cumulant, if present, is the unnormalised excess
        // kurtosis. If there are more than four cumulants, the extra
        // cumulants are ignored.
        //
        // If only two cumulants are provided, the subsequent cumulants
        // are assumed to be zero (the resulting distribution is a Gaussian).
        // If three cumulants are provided, the fourth cumulant is picked
        // automatically using the maximum entropy principle (the resulting
        // distribution is the maximum entropy S_u).
        */
        explicit JohnsonSystem(const std::vector<double>& cumulants);

        JohnsonSystem(const JohnsonSystem&);
        JohnsonSystem& operator=(const JohnsonSystem&);

        virtual ~JohnsonSystem() override;

        inline virtual JohnsonSystem* clone() const override
            {return new JohnsonSystem(*this);}

        inline double skewness() const {return skew_;}
        inline double kurtosis() const {return kurt_;}
        inline CurveType curveType() const {return curveType_;}
        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        inline virtual bool isUnimodal() const override
            {return fcn_->isUnimodal();}

        // The following method is here for consistency with other
        // Johnson models. It should always return "true".
        inline bool isValid() const {return !(curveType_ == INVALID);}

        inline virtual std::string classname() const override
            {return "JohnsonSystem";}

        // Return the name of the curve actually used
        std::string subclass() const;

        // If sigmaPlus != sigmaMinus, this function will build the maximum
        // entropy S_u (and the Gaussian if sigmaPlus == sigmaMinus).
        // Note that, due to code limitations, the ratio of
        // sigmaPlus/sigmaMinus should not exceed about 2.06 (and,
        // correspondingly, should not be less than about 1.0/2.06).
        static std::unique_ptr<JohnsonSystem> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<JohnsonSystem> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

        static CurveType select(double skewness, double kurtosis);

        // Calculate maximum entropy kurtosis for the given
        // skewness
        static double slowMaxEntKurtosis(double skewness);

        // Approximate maximum entropy kurtosis for the given
        // skewness. Works by polynomial approximation for some
        // limited skewness range and will call the slow function
        // for large skewness arguments.
        static double approxMaxEntKurtosis(double skewness);

    private:
        static const double maxAutoSkew_;

        void initialize();

        inline virtual double unscaledDensity(const double x) const override
            {return fcn_->density(x);}
        inline virtual double unscaledDensityDerivative(const double x) const override
            {return fcn_->densityDerivative(x);}
        inline virtual double unscaledCdf(const double x) const override
            {return fcn_->cdf(x);}
        inline virtual double unscaledExceedance(const double x) const override
            {return fcn_->exceedance(x);}
        inline virtual double unscaledQuantile(const double x) const override
            {return fcn_->quantile(x);}
        inline virtual double unscaledInvExceedance(const double x) const override
            {return fcn_->invExceedance(x);}
        inline virtual double unscaledCumulant(const unsigned n) const override
            {return fcn_->cumulant(n);}
        inline virtual double unscaledMode() const override
            {return fcn_->mode();}
        inline virtual double unscaledRandom(AbsRNG& gen) const override
            {return fcn_->random(gen);}
        double unscaledEntropy() const;

        AbsLocationScaleFamily* fcn_;
        double skew_;
        double kurt_;
        CurveType curveType_;

#ifdef SWIG
    public:
        inline static JohnsonSystem* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static JohnsonSystem* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /** Edgeworth expansion using just three leading cumulants */
    class EdgeworthExpansion3 : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        // The problem with this simple Edgeworth expansion is that
        // the density becomes negative for sufficiently large
        // arguments. We will check that the density is positive on
        // the interval
        // (mean - safeSigmaRange*sigma, mean + safeSigmaRange*sigma).
        // If it is not, std::invalid_argument will be thrown.
        // Negative or zero values of "safeSigmaRange" parameter
        // will disable this check. The value of "safeSigmaRange" is
        // a tunable class parameter with the default value of 3.0.
        // For this value, the maximum allowed magnitude of
        // normalized skewness is 1/3.
        //
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. If there are more than
        // three cumulants, the excess cumulants are ignored.
        explicit EdgeworthExpansion3(const std::vector<double>& cumulants);

        EdgeworthExpansion3(double mean, double standardDeviation,
                            double normalizedSkewness);

        inline virtual EdgeworthExpansion3* clone() const override
            {return new EdgeworthExpansion3(*this);}

        inline virtual ~EdgeworthExpansion3() override {}

        inline virtual bool isNonNegative() const override {return false;}

        inline double skewness() const {return skew_;}
        inline double safeSigmaRange() const {return safeRange_;}

        inline virtual std::string classname() const override
            {return "EdgeworthExpansion3";}

        static std::unique_ptr<EdgeworthExpansion3> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        // Static methods dealing with the safe sigma range
        inline static double classSafeSigmaRange() {return classSigmaRange_;}
        static void setClassSafeSigmaRange(double range);
        static void restoreDefaultSafeSigmaRange();

        // Largest skewness allowed for the given safe range
        static double largestSkewAllowed(double range);

    private:
        static double classSigmaRange_;

        void validateRange();
        double densityFactor(double x) const;
        double densityFactorDerivative(double x) const;
        double cdfFactor(double x) const;

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;

        double skew_;
        double safeRange_;
        Gaussian g_;

#ifdef SWIG
    public:
        inline static EdgeworthExpansion3* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }
#endif
    };

    /** Fechner distribution (a.k.a. split normal) */
    class FechnerDistribution : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        /**
        // Note that parameters of this constructor do not coincide
        // with the median and quantile-based sigmas
        */
        FechnerDistribution(double mu, double sigmaPlus, double sigmaMinus);

        explicit FechnerDistribution(const std::vector<double>& cumulants);

        inline virtual FechnerDistribution* clone() const override
            {return new FechnerDistribution(*this);}

        inline virtual ~FechnerDistribution() override {}

        inline double sigmaPlus() const {return sigmaPlus_;}
        inline double sigmaMinus() const {return sigmaMinus_;}

        inline virtual std::string classname() const override
            {return "FechnerDistribution";}

        static std::unique_ptr<FechnerDistribution> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);

        static std::unique_ptr<FechnerDistribution> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            double deltaLnL=0.5);

    private:
        // The following function calculates unscaled skewness
        // as a function of ratio of deltas
        static double skewFcn(double r);

        void initialize();
        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        inline virtual double unscaledMode() const override
            {return 0.0;}
        virtual double unscaledDescentDelta(bool isToTheRight,
                                            double deltaLnL) const;

        double sigmaPlus_;
        double sigmaMinus_;

        // Actual parameters of the unscaled distribution
        double sPstd_;      // Both sPstd_ and sMstd_ are positive,
        double sMstd_;      // and their sum equals 2.0.
        double xmin_;
        double xmax_;
        Gaussian g_;

        static const double largestSkew_;

#ifdef SWIG
    public:
        inline static FechnerDistribution* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static FechnerDistribution* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    /** Empirical distribution generated by a sample of unweighted points */
    class EmpiricalDistribution : public AbsDistributionModel1D
    {
    public:
        static const bool isFullOPAT = false;

        EmpiricalDistribution(const std::vector<double>& sample,
                              bool isAlreadySorted = false);

        inline virtual EmpiricalDistribution* clone() const override
            {return new EmpiricalDistribution(*this);}

        inline virtual ~EmpiricalDistribution() override {}

        // Note that calling the density, mode, or descentDelta
        // functions will always throw std::runtime_error.
        // We are not performing density estimation here.
        virtual double density(double x) const override;
        virtual double densityDerivative(double x) const override;
        virtual double mode() const override;
        virtual double descentDelta(bool isToTheRight,
                                    double deltaLnL=0.5) const override;

        virtual double cdf(double x) const override;
        virtual double exceedance(double x) const override;
        virtual double quantile(double x) const override;
        virtual double invExceedance(double x) const override;
        virtual double cumulant(unsigned n) const override;

        inline virtual std::string classname() const override
            {return "EmpiricalDistribution";}

        // The "random" function performs resampling with replacement
        virtual double random(AbsRNG& gen) const override;

        inline unsigned long sampleSize() const
            {return sortedSample_.size();}

        inline double coordinate(const unsigned long i) const
            {return sortedSample_.at(i);}

        inline double minCoordinate() const
            {return sortedSample_[0];}

        inline double maxCoordinate() const
            {return sortedSample_.back();}

        inline const std::vector<double>& sample() const
            {return sortedSample_;}

    private:
        std::vector<double> sortedSample_;
        double cumulants_[5];
    };

    /** Uniform distribution */
    class UniformDistribution : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        UniformDistribution(double location, double scale);

        /**
        // The vector of cumulants must have the size of at least two.
        // If there are more than two cumulants, the excess cumulants
        // are ignored.
        */
        explicit UniformDistribution(const std::vector<double>& cumulants);

        inline virtual UniformDistribution* clone() const override
            {return new UniformDistribution(*this);}

        inline virtual ~UniformDistribution() override {}

        inline virtual bool isUnimodal() const override
            {return false;}

        inline virtual std::string classname() const override
            {return "UniformDistribution";}

    private:
        virtual double unscaledDensity(double x) const override;
        inline virtual double unscaledDensityDerivative(double /* x */) const override
            {return 0.0;}
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledMode() const override;
        inline virtual double unscaledRandom(AbsRNG& gen) const override
            {return gen();}
    };

    /** Exponential distribution */
    class ExponentialDistribution : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        ExponentialDistribution(double location, double scale);

        /**
        // The vector of cumulants must have the size of at least two.
        // If there are more than two cumulants, the excess cumulants
        // are ignored.
        */
        explicit ExponentialDistribution(const std::vector<double>& cumulants);

        inline virtual ExponentialDistribution* clone() const override
            {return new ExponentialDistribution(*this);}

        inline virtual ~ExponentialDistribution() override {}

        inline virtual std::string classname() const override
            {return "ExponentialDistribution";}

    private:
        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        inline virtual double unscaledDescentDelta(
            const bool isToTheRight, const double deltaLnL) const override
            {return isToTheRight ? deltaLnL : 0.0;}
        inline virtual double unscaledMode() const override
            {return 0.0;}
    };
}

#endif // ASE_DISTRIBUTIONMODELS1D_HH_
