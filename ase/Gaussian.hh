#ifndef ASE_GAUSSIAN_HH_
#define ASE_GAUSSIAN_HH_

#include <vector>
#include <memory>

#include "ase/AbsLocationScaleFamily.hh"

namespace ase {
    /** Gaussian (or normal) distribution */
    class Gaussian : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        Gaussian(double mu, double sigma);

        /**
        // The vector of cumulants must have the size of at least two.
        // The first element of the vector (with index 0) is the mean
        // and the second is the variance. If there are more than
        // two cumulants, the excess cumulants are ignored.
        */
        explicit Gaussian(const std::vector<double>& cumulants);

        inline virtual Gaussian* clone() const override
            {return new Gaussian(*this);}

        inline virtual ~Gaussian() override {}

        inline virtual std::string classname() const override
            {return "Gaussian";}

        inline virtual double qWidth() const override
            {return scale();}

        inline virtual double qAsymmetry() const override
            {return 0.0;}

        inline double entropy() const
            {return log(scale()) + unscaledEntropy();}

        //@{
        /**
        // The following function will throw an exception unless
        // sigmaPlus == sigmaMinus
        */
        static std::unique_ptr<Gaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus);
        //@}

    private:
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
        virtual double unscaledDescentDelta(bool isToTheRight,
                                            double deltaLnL) const;
        double unscaledEntropy() const;

        static const double xmin_;
        static const double xmax_;

#ifdef SWIG
    public:
        inline static Gaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }
#endif
    };
}

#endif // ASE_GAUSSIAN_HH_
