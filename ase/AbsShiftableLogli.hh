#ifndef ASE_ABSSHIFTABLELOGLI_HH_
#define ASE_ABSSHIFTABLELOGLI_HH_

#include "ase/AbsLogLikelihoodCurve.hh"

namespace ase {
    /**
    // This class handles horizontal shifting and vertical scaling
    // of various built-in log-likelihood curves
    */
    class AbsShiftableLogli : public AbsLogLikelihoodCurve
    {
    public:
        AbsShiftableLogli(double shift);

        inline virtual ~AbsShiftableLogli() override {}

        virtual AbsShiftableLogli* clone() const override = 0;

        inline double shift() const {return shift_;}
        inline double factor() const {return factor_;}

        inline void setShift(const double s) {shift_ = s;}
        inline void setFactor(const double f) {factor_ = f;}

        double parMin() const override;

        double parMax() const override;

        inline double location() const override
            {return shift_ + uLocation();}

        virtual double stepSize() const override = 0;

        double maximum() const override;

        double argmax() const override;

        inline double operator()(const double p) const override
            {return factor_*uValue(p - shift_);}

        inline double derivative(const double p) const override
            {return factor_*uDerivative(p - shift_);}

        inline double secondDerivative(
            const double p, const double step = 0.0) const override
            {return factor_*uSecondDerivative(p - shift_, step);}

        virtual std::string classname() const override = 0;

        inline AbsLogLikelihoodCurve& operator*=(const double c) override
            {factor_ *= c; return *this;}

        double sigmaPlus(double deltaLogLikelihood = 0.5,
                         double stepFactor = 1.1) const override;

        double sigmaMinus(double deltaLogLikelihood = 0.5,
                          double stepFactor = 1.1) const override;
    protected:
        static void validateSigmas(const char* where,
                                   double sigmaPlus, double sigmaMinus);
        double smoothUnnormalizedMoment(
            double p0, unsigned n, double maxDeltaLogli) const;

    private:
        virtual double uParMin() const = 0;
        virtual double uParMax() const = 0;
        virtual double uLocation() const = 0;
        virtual double uMaximum() const = 0;
        virtual double uArgmax() const = 0;
        virtual double uValue(double p) const = 0;
        virtual double uDerivative(double p) const = 0;
        virtual double uSecondDerivative(double p, double step) const;
        virtual double uSigmaPlus(double deltaLogli, double f) const = 0;
        virtual double uSigmaMinus(double deltaLogli, double f) const = 0;

        double shift_;
        double factor_;
    };
}

#endif // ASE_ABSSHIFTABLELOGLI_HH_
