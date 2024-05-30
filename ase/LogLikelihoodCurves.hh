#ifndef ASE_LOGLIKELIHOODCURVES_HH_
#define ASE_LOGLIKELIHOODCURVES_HH_

#include <cfloat>
#include <utility>

#include "ase/AbsShiftableLogli.hh"
#include "ase/GeneralisedPoissonHelper.hh"
#include "ase/Poly1D.hh"
#include "ase/DistributionModel1DCopy.hh"
#include "ase/DoubleCubicLogspace.hh"
#include "ase/QuinticLogspace.hh"

// The terminology used to name the classes in this header follows,
// more or less, arXiv:physics/0406120v1
namespace ase {
    class AbsDistributionModel1D;

    double moldingVarianceAt0(double sigmaPlus, double sigmaMinus,
                              unsigned denomPower);

    // Symmetric parabola. Symmetrization is performed by assuming
    // flat prior for the parameter and using Fechner distribution
    // to calculate the mean and the width.
    class SymmetrizedParabola : public AbsShiftableLogli
    {
    public:
        SymmetrizedParabola(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~SymmetrizedParabola() override {}

        inline virtual SymmetrizedParabola* clone() const override
            {return new SymmetrizedParabola(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "SymmetrizedParabola";}

        inline double posteriorMean() const override
            {return argmax();}

        inline double posteriorVariance() const override
            {return sigPlus_*sigPlus_;}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
    };

    class BrokenParabola : public AbsShiftableLogli
    {
    public:
        BrokenParabola(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~BrokenParabola() override {}

        inline virtual BrokenParabola* clone() const override
            {return new BrokenParabola(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "BrokenParabola";}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
    };

    // This is a cubic log-likelihood from arXiv:physics/0406120v1
    // with the truncated range in x so that the part of the support
    // where the cubic starts increasing towards positive infinity
    // is removed. Note that the ratio sigmaPlus/sigmaMinus must be
    // between 1/2 and 2 (otherwise the curve reaches the value of
    // -1/2 on the side with larger sigma at a different point,
    // smaller in magnitude than that sigma).
    class TruncatedCubicLogli : public AbsShiftableLogli
    {
    public:
        TruncatedCubicLogli(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~TruncatedCubicLogli() override {}

        inline virtual TruncatedCubicLogli* clone() const override
            {return new TruncatedCubicLogli(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "TruncatedCubicLogli";}

    protected:
        inline virtual double unnormalizedMoment(
            const double p0, const unsigned n,
            const double maxDeltaLogli) const override
            {return smoothUnnormalizedMoment(p0, n, maxDeltaLogli);}

    private:
        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double alpha_;
        double beta_;
        double pmin_;
        double pmax_;
        double truncationLogli_;
    };

    class LogarithmicLogli : public AbsShiftableLogli
    {
    public:
        LogarithmicLogli(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~LogarithmicLogli() override {}

        inline virtual LogarithmicLogli* clone() const override
            {return new LogarithmicLogli(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "LogarithmicLogli";}

    protected:
        inline virtual double unnormalizedMoment(
            const double p0, const unsigned n,
            const double maxDeltaLogli) const override
            {return smoothUnnormalizedMoment(p0, n, maxDeltaLogli);}

    private:
        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double logbeta_;
        double gamma_;
        double pmin_;
        double pmax_;
    };

    class GeneralisedPoisson : public AbsShiftableLogli
    {
    public:
        inline GeneralisedPoisson(const double i_location,
                                  const double i_sigmaPlus,
                                  const double i_sigmaMinus)
            : AbsShiftableLogli(i_location),
              hlp_(0.0, i_sigmaPlus, i_sigmaMinus),
              mirror_(i_sigmaMinus > i_sigmaPlus) {}

        inline virtual ~GeneralisedPoisson() override {}

        inline virtual GeneralisedPoisson* clone() const override
            {return new GeneralisedPoisson(*this);}

        inline virtual double stepSize() const override
            {return hlp_.stepSize();}

        inline virtual std::string classname() const override
            {return "GeneralisedPoisson";}

        inline virtual double posteriorMean() const override
            {return (mirror_ ? -1.0 : 1.0)*hlp_.posteriorMean() + shift();}

        inline virtual double posteriorVariance() const override
            {return hlp_.posteriorVariance();}

    private:
        inline virtual double uParMin() const override
            {return mirror_ ? -hlp_.uParMax() : hlp_.uParMin();}
        inline virtual double uParMax() const override
            {return mirror_ ? -hlp_.uParMin() : hlp_.uParMax();}
        inline virtual double uLocation() const override
            {return mirror_ ? -hlp_.uLocation() : hlp_.uLocation();}
        inline virtual double uMaximum() const override
            {return hlp_.uMaximum();}
        inline virtual double uArgmax() const override
            {return mirror_ ? -hlp_.uArgmax() : hlp_.uArgmax();}
        inline virtual double uValue(const double p) const override
            {const double p1 = mirror_ ? -p : p; return hlp_.uValue(p1);}
        inline virtual double uDerivative(const double p) const override
            {return mirror_ ? -hlp_.uDerivative(-p) : hlp_.uDerivative(p);}
        inline virtual double uSecondDerivative(const double p, const double s) const override
            {const double p1 = mirror_ ? -p : p; return hlp_.uSecondDerivative(p1, s);}
        inline virtual double uSigmaPlus(const double d, const double f) const override
            {return mirror_ ? hlp_.uSigmaMinus(d, f) : hlp_.uSigmaPlus(d, f);}
        inline virtual double uSigmaMinus(const double d, const double f) const override
            {return mirror_ ? hlp_.uSigmaPlus(d, f) : hlp_.uSigmaMinus(d, f);}

        GeneralisedPoissonHelper hlp_;
        bool mirror_;
    };

    class ConstrainedQuartic : public AbsShiftableLogli
    {
    public:
        ConstrainedQuartic(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConstrainedQuartic() override {}

        inline virtual ConstrainedQuartic* clone() const override
            {return new ConstrainedQuartic(*this);}

        inline double alpha() const {return alpha_;}
        inline double beta() const {return beta_;}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "ConstrainedQuartic";}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double alpha_;
        double beta_;
    };

    class MoldedQuartic : public AbsShiftableLogli
    {
    public:
        MoldedQuartic(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~MoldedQuartic() override {}

        inline virtual MoldedQuartic* clone() const override
            {return new MoldedQuartic(*this);}

        inline double a() const {return a_;}
        inline double b() const {return b_;}
        inline double c() const {return c_;}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "MoldedQuartic";}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        // The shape is -1/2 (a x^4 + b x^3 + c x^2)
        double a_;
        double b_;
        double c_;
    };

    class MatchedQuintic : public AbsShiftableLogli
    {
    public:
        MatchedQuintic(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~MatchedQuintic() override {}

        inline virtual MatchedQuintic* clone() const override
            {return new MatchedQuintic(*this);}

        inline double a() const {return a_;}
        inline double b() const {return b_;}
        inline double c() const {return c_;}
        inline double d() const {return d_;}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "MatchedQuintic";}

    private:
        double quinticValue(double p) const;
        double quinticDeriv(double p) const;
        double quinticSecondDeriv(double p) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        // The shape is -1/2 (a x^5 + b x^4 + c x^3 + d x^2) on the interval
        // [-sigMinus_, sigPlus_], extrapolated with quadratic beyond this
        // interval.
        double a_;
        double b_;
        double c_;
        double d_;

        double dplus_;
        double dminus_;
    };

    // The following class remains abstract
    class DoubleQuartic : public AbsShiftableLogli
    {
    public:
        DoubleQuartic(double location, double sigmaPlus, double sigmaMinus,
                      double effectiveVarianceAt0);

        inline virtual ~DoubleQuartic() override {}

        virtual AbsShiftableLogli* clone() const override = 0;

        virtual double stepSize() const override;

        virtual std::string classname() const override = 0;

    private:
        double quarticValue(double p) const;
        double quarticDeriv(double p) const;
        double quarticSecondDeriv(double p) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        // The shape is -1/2 (a x^4 + b x^3 + c x^2) on the interval
        // [0, sigPlus_], extrapolated with quadratic beyond this interval.
        // Same happens for the interval [-sigMinus_, 0]. The second derivative
        // is continuous at 0, so that the coefficient c is the same for
        // both sides.
        double aleft_;
        double bleft_;
        double aright_;
        double bright_;
        double c_;

        double dplus_;
        double dminus_;
    };

    class MoldedDoubleQuartic : public DoubleQuartic
    {
    public:
        inline MoldedDoubleQuartic(const double location,
                                   const double sigmaPlus,
                                   const double sigmaMinus)
            : DoubleQuartic(location, sigmaPlus, sigmaMinus,
                            moldingVarianceAt0(sigmaPlus, sigmaMinus, 2U)) {}

        inline virtual ~MoldedDoubleQuartic() override {}

        inline virtual MoldedDoubleQuartic* clone() const override
            {return new MoldedDoubleQuartic(*this);}

        inline virtual std::string classname() const override
            {return "MoldedDoubleQuartic";}
    };

    class SimpleDoubleQuartic : public DoubleQuartic
    {
    public:
        inline SimpleDoubleQuartic(const double location,
                                   const double sigmaPlus,
                                   const double sigmaMinus)
            : DoubleQuartic(location, sigmaPlus, sigmaMinus, sigmaPlus*sigmaMinus) {}

        inline virtual ~SimpleDoubleQuartic() override {}

        inline virtual SimpleDoubleQuartic* clone() const override
            {return new SimpleDoubleQuartic(*this);}

        inline virtual std::string classname() const override
            {return "SimpleDoubleQuartic";}
    };

    // The following class remains abstract
    class DoubleQuintic : public AbsShiftableLogli
    {
    public:
        DoubleQuintic(double location, double sigmaPlus, double sigmaMinus,
                      double effectiveVarianceAt0);

        inline virtual ~DoubleQuintic() override {}

        virtual AbsShiftableLogli* clone() const override = 0;

        virtual double stepSize() const override;

        virtual std::string classname() const override = 0;

    private:
        double quinticValue(double p) const;
        double quinticDeriv(double p) const;
        double quinticSecondDeriv(double p) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        // The shape is -1/2 (a x^5 + b x^4 + c x^3 + d x^2) on the interval
        // [0, sigPlus_], extrapolated with quadratic beyond this interval.
        // Same happens for the interval [-sigMinus_, 0]. The second derivative
        // is continuous at 0, so that the coefficient d is the same for
        // both sides.
        double aleft_;
        double bleft_;
        double cleft_;
        double aright_;
        double bright_;
        double cright_;
        double d_;

        double dplus_;
        double dminus_;
    };

    class MoldedDoubleQuintic : public DoubleQuintic
    {
    public:
        inline MoldedDoubleQuintic(const double location,
                                   const double sigmaPlus,
                                   const double sigmaMinus)
            : DoubleQuintic(location, sigmaPlus, sigmaMinus,
                            moldingVarianceAt0(sigmaPlus, sigmaMinus, 2U)) {}

        inline virtual ~MoldedDoubleQuintic() override {}

        inline virtual MoldedDoubleQuintic* clone() const override
            {return new MoldedDoubleQuintic(*this);}

        inline virtual std::string classname() const override
            {return "MoldedDoubleQuintic";}
    };

    class SimpleDoubleQuintic : public DoubleQuintic
    {
    public:
        inline SimpleDoubleQuintic(const double location,
                                   const double sigmaPlus,
                                   const double sigmaMinus)
            : DoubleQuintic(location, sigmaPlus, sigmaMinus, sigmaPlus*sigmaMinus) {}

        inline virtual ~SimpleDoubleQuintic() override {}

        inline virtual SimpleDoubleQuintic* clone() const override
            {return new SimpleDoubleQuintic(*this);}

        inline virtual std::string classname() const override
            {return "SimpleDoubleQuintic";}
    };

    class Interpolated7thDegree : public AbsShiftableLogli
    {
    public:
        Interpolated7thDegree(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~Interpolated7thDegree() override {}

        inline virtual Interpolated7thDegree* clone() const override
            {return new Interpolated7thDegree(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "Interpolated7thDegree";}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        Poly1D septic_;
        Poly1D deriv_;
        Poly1D secondDeriv_;
    };

    class VariableSigmaLogli : public AbsShiftableLogli
    {
    public:
        VariableSigmaLogli(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~VariableSigmaLogli() override {}

        inline virtual VariableSigmaLogli* clone() const override
            {return new VariableSigmaLogli(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "VariableSigmaLogli";}

    protected:
        inline virtual double unnormalizedMoment(
            const double p0, const unsigned n,
            const double maxDeltaLogli) const override
            {return smoothUnnormalizedMoment(p0, n, maxDeltaLogli);}

    private:
        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double sigma0_;
        double sigmaPrime_;
        double pmin_;
        double pmax_;
        double truncationLogli_;
    };

    class VariableVarianceLogli : public AbsShiftableLogli
    {
    public:
        VariableVarianceLogli(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~VariableVarianceLogli() override {}

        inline virtual VariableVarianceLogli* clone() const override
            {return new VariableVarianceLogli(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "VariableVarianceLogli";}

    protected:
        inline virtual double unnormalizedMoment(
            const double p0, const unsigned n,
            const double maxDeltaLogli) const override
            {return smoothUnnormalizedMoment(p0, n, maxDeltaLogli);}

    private:
        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double v0_;
        double vPrime_;
        double pmin_;
        double pmax_;
        double truncationLogli_;
    };

    class VariableLogSigma : public AbsShiftableLogli
    {
    public:
        VariableLogSigma(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~VariableLogSigma() override {}

        inline virtual VariableLogSigma* clone() const override
            {return new VariableLogSigma(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "VariableLogSigma";}

    private:
        double sigmaValue(double x) const;
        void sigmaDerivative(double x, double* value, double* deriv) const;
        void sigmaSecondDerivative(double x, double* value,
                                   double* deriv, double* secondDeriv) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double logPlus_;
        double logMinus_;
    };

    // The following class remains abstract
    class DoubleCubicLogSigma : public AbsShiftableLogli
    {
    public:
        DoubleCubicLogSigma(double location, double sigmaPlus, double sigmaMinus,
                            double effectiveSigmaAt0);

        inline virtual ~DoubleCubicLogSigma() override {}

        inline virtual DoubleCubicLogSigma* clone() const override = 0;

        virtual double stepSize() const override;

        inline virtual std::string classname() const override = 0;

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        DoubleCubicLogspace interp_;
    };

    class MoldedCubicLogSigma : public DoubleCubicLogSigma
    {
    public:
        MoldedCubicLogSigma(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~MoldedCubicLogSigma() override {}

        inline virtual MoldedCubicLogSigma* clone() const override
            {return new MoldedCubicLogSigma(*this);}

        inline virtual std::string classname() const override
            {return "MoldedCubicLogSigma";}

        static double getEffectiveSigmaAt0(double sigmaPlus, double sigmaMinus);
    };

    class QuinticLogSigma : public AbsShiftableLogli
    {
    public:
        QuinticLogSigma(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~QuinticLogSigma() override {}

        inline virtual QuinticLogSigma* clone() const override
            {return new QuinticLogSigma(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const
            {return "QuinticLogSigma";}

    private:
        // Check if the unshifted curve has another extremum on the
        // interval [-sigmaMinus, sigmaPlus] apart from the one at 0
        bool hasExtraExtremum() const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        QuinticLogspace interp_;
    };

    class PDGLogli : public AbsShiftableLogli
    {
    public:
        PDGLogli(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~PDGLogli() override {}

        inline virtual PDGLogli* clone() const override
            {return new PDGLogli(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "PDGLogli";}

    private:
        double sigmaValue(double x) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double sigma0_;
        double sigmaPrime_;
    };

    class LogLogisticBeta : public AbsShiftableLogli
    {
    public:
        LogLogisticBeta(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~LogLogisticBeta() override {}

        inline virtual LogLogisticBeta* clone() const override
            {return new LogLogisticBeta(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "LogLogisticBeta";}

    private:
        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;
        double a_;
        double c_;
        double g_;
        SymmetrizedParabola symp_;
    };

    class DistributionLogli : public AbsShiftableLogli
    {
    public:
        // If the parameter "minDensity" is not positive, some
        // reasonable default value will be chosen automatically.
        // "minDensity" affects the log likelihood curve support
        // in the parameter space.
        DistributionLogli(const AbsDistributionModel1D& distro,
                          double x0, double minDensity = 0.0);

        inline virtual ~DistributionLogli() override {}

        inline virtual DistributionLogli* clone() const override
            {return new DistributionLogli(*this);}

        inline virtual double stepSize() const override
            {return stepSize_;}

        inline virtual std::string classname() const override
            {return "DistributionLogli";}

        inline double x0() const
            {return x_;}

        inline const AbsDistributionModel1D& distribution() const
            {return distro_;}

        inline double posteriorMean() const override
            {return x_ - distro_.cumulant(1) + shift();}

        inline double posteriorVariance() const override
            {return distro_.cumulant(2);}

    private:
        static const double defaultDensityCutoff_;

        void adjustParameterLimits(double densityCutoff);

        inline double pDensity(const double p) const
            {return distro_.density(x_ - p);}

        inline double pDensityDerivative(const double p) const
            {return -distro_.densityDerivative(x_ - p);}

        double pDensitySecondDerivative(double p, double step) const;

        inline virtual double uParMin() const override {return pmin_;}
        inline virtual double uParMax() const override {return pmax_;}
        virtual double uLocation() const override;
        inline virtual double uMaximum() const override {return 0.0;}
        virtual double uArgmax() const override;
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        DistributionModel1DCopy distro_;
        double x_;
        double pmin_;
        double pmax_;
        double stepSize_;
        double mode_;
        double logAtMode_;
    };

    class ConservativeSpline : public AbsShiftableLogli
    {
    public:
        ConservativeSpline(double location, double sigmaPlus,
                           double sigmaMinus, double secondDerivLimitFactor);

        inline virtual ~ConservativeSpline() override {}

        inline virtual ConservativeSpline* clone() const override
            {return new ConservativeSpline(*this);}

        virtual double stepSize() const override;

        inline virtual std::string classname() const override
            {return "ConservativeSpline";}

        inline double limitingFactor() const
            {return ksq_;}

        // Maximum possible value for the "secondDerivLimitFactor"
        // constructor argument
        static double maxDerivLimitFactor(double sigmaPlus, double sigmaMinus);

    protected:
        virtual double unnormalizedMoment(
            double p0, unsigned n, double maxDeltaLogli) const override;

    private:
        static const double tol_;

        static double sqrformula(double ksq, double s0sq, double sp);
        static bool isKsqUsable(double sp, double sm, double ksq);
        static double sqrtDerivLimitFactor(double sigmaPlus, double sigmaMinus);
        static double aformula(double ksq, double s0sq, double sp, bool posRoot);
        static double rxformula(double ksq, double s0sq, double sp, bool posRoot);
        static double cformula(double ksq, double s0sq, double sp, bool posRoot);

        // The following static functions need ksq >= 1
        static std::pair<double,double> s0sqRange(
            double sigmaPlus, double sigmaMinus, double ksq);
        static double adelta(double ksq, double s0sq, double sp, double sm,
                             bool posRootForSp, bool posRootForSm);

        double getFactor(bool isRight) const;

        inline virtual double uParMin() const override {return -DBL_MAX;}
        inline virtual double uParMax() const override {return DBL_MAX;}
        inline virtual double uLocation() const override {return 0.0;}
        inline virtual double uMaximum() const override {return 0.0;}
        inline virtual double uArgmax() const override {return 0.0;}
        virtual double uValue(double p) const override;
        virtual double uDerivative(double p) const override;
        virtual double uSecondDerivative(double p, double step) const override;
        virtual double uSigmaPlus(double deltaLogli, double f) const override;
        virtual double uSigmaMinus(double deltaLogli, double f) const override;

        double sigPlus_;
        double sigMinus_;

        double ksq_;
        double s0sq_;
        double a_;
        double c_;
        double d_;
        double rx0_;
        double lx0_;

        class ConservativeSplineADelta
        {
        public:
            inline ConservativeSplineADelta(const double sp, const double sm,
                                            const double ksq, const bool posRootForSp,
                                            const bool posRootForSm)
                : sp_(sp), sm_(sm), ksq_(ksq),
                  posRootForSp_(posRootForSp), posRootForSm_(posRootForSm) {}

            inline double operator()(const double s0sq) const
            {
                return ase::ConservativeSpline::adelta(
                    ksq_, s0sq, sp_, sm_, posRootForSp_, posRootForSm_);
            }

        private:
            double sp_;
            double sm_;
            double ksq_;
            bool posRootForSp_;
            bool posRootForSm_;
        };
        friend class ConservativeSplineADelta;
    };

    // Several convenience classes inheriting from ConservativeSpline
    // with standard 3-argument constructors. The class below, for example,
    // makes sure that the sigma for large parameter arguments does not
    // differ from sigma^+ or sigma^- (depending on direction) by more than 5%.
    class ConservativeSigma05 : public ConservativeSpline
    {
    public:
        ConservativeSigma05(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConservativeSigma05() override {}

        inline virtual ConservativeSigma05* clone() const override
            {return new ConservativeSigma05(*this);}

        inline virtual std::string classname() const override
            {return "ConservativeSigma05";}
    };

    class ConservativeSigma10 : public ConservativeSpline
    {
    public:
        ConservativeSigma10(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConservativeSigma10() override {}

        inline virtual ConservativeSigma10* clone() const override
            {return new ConservativeSigma10(*this);}

        inline virtual std::string classname() const override
            {return "ConservativeSigma10";}
    };

    class ConservativeSigma15 : public ConservativeSpline
    {
    public:
        ConservativeSigma15(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConservativeSigma15() override {}

        inline virtual ConservativeSigma15* clone() const override
            {return new ConservativeSigma15(*this);}

        inline virtual std::string classname() const override
            {return "ConservativeSigma15";}
    };

    class ConservativeSigma20 : public ConservativeSpline
    {
    public:
        ConservativeSigma20(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConservativeSigma20() override {}

        inline virtual ConservativeSigma20* clone() const override
            {return new ConservativeSigma20(*this);}

        inline virtual std::string classname() const override
            {return "ConservativeSigma20";}
    };

    class ConservativeSigmaMax : public ConservativeSpline
    {
    public:
        ConservativeSigmaMax(double location, double sigmaPlus, double sigmaMinus);

        inline virtual ~ConservativeSigmaMax() override {}

        inline virtual ConservativeSigmaMax* clone() const override
            {return new ConservativeSigmaMax(*this);}

        inline virtual std::string classname() const override
            {return "ConservativeSigmaMax";}
    };
}

#endif // ASE_LOGLIKELIHOODCURVES_HH_
