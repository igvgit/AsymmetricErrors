#ifndef ASE_INTERPOLATEDDENSITY1D_HH_
#define ASE_INTERPOLATEDDENSITY1D_HH_

#include <vector>
#include <stdexcept>

#include "ase/AbsLocationScaleFamily.hh"
#include "ase/EquidistantGrid.hh"

namespace ase {
    /**
    // Distribution specified by its discretized density curve.
    // The curve is interpolated using cubic Hermite splines.
    */
    class InterpolatedDensity1D : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        InterpolatedDensity1D(double unscaledXmin, double unscaledXmax,
                              const std::vector<double>& unscaledDensityValues);

        InterpolatedDensity1D(double unscaledXmin, double unscaledXmax,
                              const std::vector<double>& unscaledDensityValues,
                              const std::vector<double>& unscaledDensityDerivs);

        template <class Functor>
        inline InterpolatedDensity1D(const double unscaledXmin,
                                     const double unscaledXmax,
                                     const unsigned nScanPoints,
                                     const Functor& unscaledDensityFcn,
                                     const double derivativeEpsilon)
            : AbsLocationScaleFamily(0.0, 1.0),
              grid_(nScanPoints, unscaledXmin, unscaledXmax),
              values_(nScanPoints),
              derivatives_(nScanPoints),
              cumulantsCalculated_(false),
              entropyCalculated_(false),
              unimodalityDetermined_(false)
        {
            if (!(derivativeEpsilon > 0.0)) throw std::invalid_argument(
                "In ase::InterpolatedDensity1D constructor: "
                "parameter derivativeEpsilon must be positive");
            for (unsigned i=0; i<nScanPoints; ++i)
            {
                const double x = grid_.coordinate(i);
                values_[i] = unscaledDensityFcn(x);
                const volatile double xplus = x + derivativeEpsilon;
                const volatile double xminus = x - derivativeEpsilon;
                const double twoh = xplus - xminus;
                derivatives_[i] = (unscaledDensityFcn(xplus) -
                                   unscaledDensityFcn(xminus))/twoh;
            }
            normalize();
        }

        template <class Functor1, class Functor2>
        inline InterpolatedDensity1D(const double unscaledXmin,
                                     const double unscaledXmax,
                                     const unsigned nScanPoints,
                                     const Functor1& unscaledDensityFcn,
                                     const Functor2& derivativeFcn)
            : AbsLocationScaleFamily(0.0, 1.0),
              grid_(nScanPoints, unscaledXmin, unscaledXmax),
              values_(nScanPoints),
              derivatives_(nScanPoints),
              cumulantsCalculated_(false),
              entropyCalculated_(false),
              unimodalityDetermined_(false)
        {
            for (unsigned i=0; i<nScanPoints; ++i)
            {
                const double x = grid_.coordinate(i);
                values_[i] = unscaledDensityFcn(x);
                derivatives_[i] = derivativeFcn(x);
            }
            normalize();
        }

        inline virtual InterpolatedDensity1D* clone() const override
            {return new InterpolatedDensity1D(*this);}

        inline virtual ~InterpolatedDensity1D() override {}

        inline virtual bool isUnimodal() const override
            {return determineUnimodality();}

        inline virtual std::string classname() const override
            {return "InterpolatedDensity1D";}

        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        inline unsigned nCoords() const {return grid_.nCoords();}

    private:
        void normalize();
        void calculateCumulants() const;
        double calculateMoment(double mu, unsigned power) const;

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        double unscaledEntropy() const;
        bool determineUnimodality() const;

        EquidistantGrid grid_;
        std::vector<double> values_;
        std::vector<double> derivatives_;
        std::vector<double> cdfValues_;
        std::vector<double> excValues_;

        mutable double cumulants_[5];
        mutable double entropy_;
        mutable bool unimodal_;
        mutable bool cumulantsCalculated_;
        mutable bool entropyCalculated_;
        mutable bool unimodalityDetermined_;
    };
}

#endif // ASE_INTERPOLATEDDENSITY1D_HH_
