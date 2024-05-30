#ifndef ASE_CUBICHERMITEINTERPOLATOREG_HH_
#define ASE_CUBICHERMITEINTERPOLATOREG_HH_

#include <vector>
#include <stdexcept>

#include "ase/AbsLogLikelihoodCurve.hh"
#include "ase/EquidistantGrid.hh"

namespace ase {
    /** Cubic Hermite interpolator on an equidistant grid */
    class CubicHermiteInterpolatorEG : public AbsLogLikelihoodCurve
    {
    public:
        CubicHermiteInterpolatorEG(double minParam, double maxParam,
                                   const std::vector<double>& values);

        CubicHermiteInterpolatorEG(double minParam, double maxParam,
                                   const std::vector<double>& values,
                                   const std::vector<double>& derivatives);

        CubicHermiteInterpolatorEG(double minParam, double maxParam,
                                   unsigned nScanPoints,
                                   const AbsLogLikelihoodCurve& curveToDiscretize);

        template <class Functor>
        inline CubicHermiteInterpolatorEG(const double minParam,
                                          const double maxParam,
                                          const unsigned nScanPoints,
                                          const Functor& valueFcn,
                                          const double derivativeEpsilon)
            : grid_(nScanPoints, minParam, maxParam),
              values_(nScanPoints),
              derivatives_(nScanPoints)
        {
            if (!(derivativeEpsilon > 0.0)) throw std::invalid_argument(
                "In ase::CubicHermiteInterpolatorEG constructor: "
                "parameter derivativeEpsilon must be positive");
            for (unsigned i=0; i<nScanPoints; ++i)
            {
                const double x = grid_.coordinate(i);
                values_[i] = valueFcn(x);
                const volatile double xplus = x + derivativeEpsilon;
                const volatile double xminus = x - derivativeEpsilon;
                const double twoh = xplus - xminus;
                derivatives_[i] = (valueFcn(xplus) - valueFcn(xminus))/twoh;
            }
            findMaximum();
            findLocation();
        }

        template <class Functor1, class Functor2>
        inline CubicHermiteInterpolatorEG(const double minParam,
                                          const double maxParam,
                                          const unsigned nScanPoints,
                                          const Functor1& valueFcn,
                                          const Functor2& derivativeFcn)
            : grid_(nScanPoints, minParam, maxParam),
              values_(nScanPoints),
              derivatives_(nScanPoints)
        {
            for (unsigned i=0; i<nScanPoints; ++i)
            {
                const double x = grid_.coordinate(i);
                values_[i] = valueFcn(x);
                derivatives_[i] = derivativeFcn(x);
            }
            findMaximum();
            findLocation();
        }

        inline virtual CubicHermiteInterpolatorEG* clone() const override
            {return new CubicHermiteInterpolatorEG(*this);}

        inline virtual ~CubicHermiteInterpolatorEG() override {}

        inline virtual double parMin() const override
            {return grid_.min();}

        inline virtual double parMax() const override
            {return grid_.max();}

        inline unsigned nPoints() const
            {return grid_.nCoords();}

        inline virtual double stepSize() const override
            {return grid_.intervalWidth(0U);}

        inline const std::vector<double>& getValues() const
            {return values_;}

        inline const std::vector<double>& getDerivatives() const
            {return derivatives_;}

        inline virtual double location() const override
            {return location_;}

        inline virtual double maximum() const override
            {return logliMax_;}

        inline virtual double argmax() const override
            {return argmax_;}

        virtual double operator()(double parameter) const override;

        virtual double derivative(double parameter) const override;

        virtual double secondDerivative(
            double parameter, double step = 0.0) const override;

        inline virtual std::string classname() const override
            {return "CubicHermiteInterpolatorEG";}

        virtual AbsLogLikelihoodCurve& operator*=(double c) override;

    protected:
        double unnormalizedMoment(
            double p0, unsigned n, double maxDeltaLogli) const override;

    private:
        void findMaximum();
        std::pair<double,double> findMinimum() const;
        void findLocation();

        EquidistantGrid grid_;
        std::vector<double> values_;
        std::vector<double> derivatives_;
        double logliMax_;
        double argmax_;
        double location_;
    };
}

#endif // ASE_CUBICHERMITEINTERPOLATOREG_HH_
