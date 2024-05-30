#ifndef ASE_TABULATEDDENSITY1D_HH_
#define ASE_TABULATEDDENSITY1D_HH_

#include <vector>

#include "ase/AbsLocationScaleFamily.hh"
#include "ase/EquidistantGrid.hh"

namespace ase {
    /**
    // Distribution specified by its discretized density curve.
    // The curve is interpolated linearly between values.
    */
    class TabulatedDensity1D : public AbsLocationScaleFamily
    {
    public:
        inline TabulatedDensity1D(const double unscaledXmin,
                                  const double unscaledXmax,
                                  const std::vector<double>& unscaledDensityValues)
            : AbsLocationScaleFamily(0.0, 1.0),
              grid_(unscaledDensityValues.size(), unscaledXmin, unscaledXmax),
              values_(unscaledDensityValues),
              cumulantsCalculated_(false),
              entropyCalculated_(false),
              unimodalityDetermined_(false)
        {
            normalize();
        }

        template <class Functor>
        inline TabulatedDensity1D(const double unscaledXmin,
                                  const double unscaledXmax,
                                  const unsigned nScanPoints,
                                  const Functor& unscaledDensityFcn)
            : AbsLocationScaleFamily(0.0, 1.0),
              grid_(nScanPoints, unscaledXmin, unscaledXmax),
              values_(nScanPoints),
              cumulantsCalculated_(false),
              entropyCalculated_(false),
              unimodalityDetermined_(false)
        {
            for (unsigned i=0; i<nScanPoints; ++i)
                values_[i] = unscaledDensityFcn(grid_.coordinate(i));
            normalize();
        }

        inline virtual TabulatedDensity1D* clone() const override
            {return new TabulatedDensity1D(*this);}

        inline virtual ~TabulatedDensity1D() override {}

        inline virtual bool isUnimodal() const override
            {return determineUnimodality();}

        inline virtual std::string classname() const override
            {return "TabulatedDensity1D";}

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
        virtual double unscaledMode() const override;
        double unscaledEntropy() const;
        bool determineUnimodality() const;

        EquidistantGrid grid_;
        std::vector<double> values_;
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

#endif // ASE_TABULATEDDENSITY1D_HH_
