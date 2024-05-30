#ifndef ASE_SYMMETRICBETAGAUSSIAN_HH_
#define ASE_SYMMETRICBETAGAUSSIAN_HH_

#include <utility>
#include <memory>
#include <sstream>
#include <cassert>

#include "ase/OPATGaussian.hh"
#include "ase/SymbetaDoubleIntegral.hh"
#include "ase/miscUtils.hh"

namespace ase {
    class SymmetricBetaGaussian : public OPATGaussian<SymbetaDoubleIntegral<long double> >
    {
    public:
        typedef OPATGaussian<SymbetaDoubleIntegral<long double> > Base;
        typedef typename Base::Transform Transform;

        /**
        // Parameters sigmaPlus and sigmaMinus can be negative.
        // If the sigma is negative, the direction of the corresponding
        // Gaussian piece will be reversed and, of course, the "center"
        // parameter might no longer be the real median.
        */
        SymmetricBetaGaussian(double center, double sigmaPlus, double sigmaMinus,
                              unsigned p, double h);

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
        // normalized skewness must be less than ?.
        */
        SymmetricBetaGaussian(const std::vector<double>& cumulants,
                              unsigned p, double h);

        inline virtual SymmetricBetaGaussian* clone() const override
            {return new SymmetricBetaGaussian(*this);}

        inline virtual ~SymmetricBetaGaussian() override {}

        inline unsigned p() const {return tr_.p();}
        inline double h() const {return tr_.h();}

        inline virtual std::string classname() const override
            {return "SymmetricBetaGaussian";}

        static std::unique_ptr<SymmetricBetaGaussian> fromQuantiles(
            double median, double sigmaPlus, double sigmaMinus,
            unsigned p, double h);

        static std::unique_ptr<SymmetricBetaGaussian> fromModeAndDeltas(
            double mode, double deltaPlus, double deltaMinus,
            unsigned p, double h, double deltaLnL=0.5);

        // The first element of the returned pair is the value of
        // r = min(sigmaPlus, sigmaMinus)/max(sigmaPlus, sigmaMinus)
        // at which the smallest possible ratio is achieved.
        // The second  element of the returned pair is that ratio.
        static std::pair<double,double> minQuantileRatio(unsigned p, double h);
        static std::pair<double,double> minDescentDeltaRatio(unsigned p, double h,
                                                             double deltaLnL=0.5);
    private:
        // Skewness for r = sigmaPlus/sigmaMinus values between 0 and 1
        static double skewnessForSmallR(double r, unsigned p, double h);

        virtual long double calculateMoment(
            long double mu, unsigned power) const override;

        class SkewFcn;
        friend class SkewFcn;
        class SkewFcn
        {
        public:
            inline SkewFcn(const unsigned p, const double h)
                : p_(p), h_(h) {}
            
            inline double operator()(const double r) const
                {return SymmetricBetaGaussian::skewnessForSmallR(r, p_, h_);}

        private:
            unsigned p_;
            double h_;
        };

#ifdef SWIG
    public:
        inline static SymmetricBetaGaussian* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus,
            const unsigned p, const double h)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus, p, h).release();
        }

        inline static SymmetricBetaGaussian* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const unsigned p, const double h, const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus,
                                     p, h, deltaLnL).release();
        }
#endif
    };

    // A convenience class for making SymmetricBetaGaussian
    // with a few fixed values of p and h parameters
    template<unsigned P, unsigned H>
    struct SymmetricBetaGaussian_p_h : public SymmetricBetaGaussian
    {
        inline SymmetricBetaGaussian_p_h(const double center,
                                         const double sigmaPlus,
                                         const double sigmaMinus)
            : SymmetricBetaGaussian(center, sigmaPlus, sigmaMinus, P, H/10.0) {}

        inline SymmetricBetaGaussian_p_h(const std::vector<double>& cumulants)
            : SymmetricBetaGaussian(cumulants, P, H/10.0) {}

        inline virtual SymmetricBetaGaussian_p_h* clone() const override
            {return new SymmetricBetaGaussian_p_h(*this);}

        inline virtual ~SymmetricBetaGaussian_p_h() override {}

        inline virtual std::string classname() const override
        {
            std::ostringstream os;
            os << "SymmetricBetaGaussian_" << P << '_' << H;
            return os.str();
        }

        inline static std::unique_ptr<SymmetricBetaGaussian_p_h> fromQuantiles(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            std::unique_ptr<SymmetricBetaGaussian> tmp =
                SymmetricBetaGaussian::fromQuantiles(
                    median, sigmaPlus, sigmaMinus, P, H/10.0);
            assert(tmp->p() == P);
            assert_approx_equal(tmp->h(), H/10.0);
            return std::unique_ptr<SymmetricBetaGaussian_p_h>(
                new SymmetricBetaGaussian_p_h(tmp->location(), tmp->sigmaPlus(),
                                              tmp->sigmaMinus()));
        }

        inline static std::unique_ptr<SymmetricBetaGaussian_p_h> fromModeAndDeltas(
            const double mode, const double deltaPlus,
            const double deltaMinus, const double deltaLnL=0.5)
        {
            std::unique_ptr<SymmetricBetaGaussian> tmp =
                SymmetricBetaGaussian::fromModeAndDeltas(
                    mode, deltaPlus, deltaMinus, P, H/10.0, deltaLnL);
            assert(tmp->p() == P);
            assert_approx_equal(tmp->h(), H/10.0);
            return std::unique_ptr<SymmetricBetaGaussian_p_h>(
                new SymmetricBetaGaussian_p_h(tmp->location(), tmp->sigmaPlus(),
                                              tmp->sigmaMinus()));
        }

        // The first element of the returned pair is the value of
        // r = min(sigmaPlus, sigmaMinus)/max(sigmaPlus, sigmaMinus)
        // at which the smallest possible ratio is achieved.
        // The second  element of the returned pair is that ratio.
        inline static std::pair<double,double> minQuantileRatio()
            {return SymmetricBetaGaussian::minQuantileRatio(P, H/10.0);}

        inline static std::pair<double,double> minDescentDeltaRatio(const double deltaLnL=0.5)
            {return SymmetricBetaGaussian::minDescentDeltaRatio(P, H/10.0, deltaLnL);}

#ifdef SWIG
        inline static SymmetricBetaGaussian_p_h* fromQuantilesBarePtr(
            const double median, const double sigmaPlus, const double sigmaMinus)
        {
            return fromQuantiles(median, sigmaPlus, sigmaMinus).release();
        }

        inline static SymmetricBetaGaussian_p_h* fromModeAndDeltasBarePtr(
            const double mode, const double deltaPlus, const double deltaMinus,
            const double deltaLnL=0.5)
        {
            return fromModeAndDeltas(mode, deltaPlus, deltaMinus, deltaLnL).release();
        }
#endif
    };

    typedef SymmetricBetaGaussian_p_h<1U,10U> SymmetricBetaGaussian_1_10;
    typedef SymmetricBetaGaussian_p_h<1U,15U> SymmetricBetaGaussian_1_15;
    typedef SymmetricBetaGaussian_p_h<1U,20U> SymmetricBetaGaussian_1_20;
    typedef SymmetricBetaGaussian_p_h<1U,25U> SymmetricBetaGaussian_1_25;
    typedef SymmetricBetaGaussian_p_h<1U,30U> SymmetricBetaGaussian_1_30;
    typedef SymmetricBetaGaussian_p_h<2U,10U> SymmetricBetaGaussian_2_10;
    typedef SymmetricBetaGaussian_p_h<2U,15U> SymmetricBetaGaussian_2_15;
    typedef SymmetricBetaGaussian_p_h<2U,20U> SymmetricBetaGaussian_2_20;
    typedef SymmetricBetaGaussian_p_h<2U,25U> SymmetricBetaGaussian_2_25;
    typedef SymmetricBetaGaussian_p_h<2U,30U> SymmetricBetaGaussian_2_30;
    typedef SymmetricBetaGaussian_p_h<3U,10U> SymmetricBetaGaussian_3_10;
    typedef SymmetricBetaGaussian_p_h<3U,15U> SymmetricBetaGaussian_3_15;
    typedef SymmetricBetaGaussian_p_h<3U,20U> SymmetricBetaGaussian_3_20;
    typedef SymmetricBetaGaussian_p_h<3U,25U> SymmetricBetaGaussian_3_25;
    typedef SymmetricBetaGaussian_p_h<3U,30U> SymmetricBetaGaussian_3_30;
    typedef SymmetricBetaGaussian_p_h<4U,10U> SymmetricBetaGaussian_4_10;
    typedef SymmetricBetaGaussian_p_h<4U,15U> SymmetricBetaGaussian_4_15;
    typedef SymmetricBetaGaussian_p_h<4U,20U> SymmetricBetaGaussian_4_20;
    typedef SymmetricBetaGaussian_p_h<4U,25U> SymmetricBetaGaussian_4_25;
    typedef SymmetricBetaGaussian_p_h<4U,30U> SymmetricBetaGaussian_4_30;
}

#endif // ASE_SYMMETRICBETAGAUSSIAN_HH_
