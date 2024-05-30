#include <numeric>
#include <stdexcept>

#include "ase/EquidistantGrid.hh"
#include "ase/DiscretizedConvolution.hh"

namespace ase {
    DiscretizedConvolution::DiscretizedConvolution(
        const AbsDistributionModel1D& m1,
        const AbsDistributionModel1D& m2,
        const double i_xmin, const double i_xmax,
        const unsigned i_nIntervals, const bool i_normalize)
        : convolution_(i_nIntervals), xmin_(i_xmin), xmax_(i_xmax),
          nIntervals_(i_nIntervals), normalized_(false)
    {
        if (xmin_ >= xmax_) throw std::invalid_argument(
            "In ase::DiscretizedConvolution constructor: "
            "xmin must be smaller than xmax");
        if (nIntervals_ < 2U) throw std::invalid_argument(
            "In ase::DiscretizedConvolution constructor: "
            "number of discretization intervals must be at least 2");

        std::vector<double> d2(nIntervals_);
        {
            const EquidistantGrid grid(nIntervals_+1U, xmin_, xmax_);
            double cdf = m2.cdf(xmin_);
            for (unsigned i=0; i<nIntervals_; ++i)
            {
                const double upper = grid.coordinate(i+1U);
                const double uCdf = m2.cdf(upper);
                d2[i] = uCdf - cdf;
                cdf = uCdf;
            }
        }

        const double h = intervalWidth();
        const unsigned fIntervals = 2U*nIntervals_ - 1U;
        std::vector<double> d1(fIntervals);
        {
            const double fXmax = fIntervals*h/2.0;
            const EquidistantGrid grid(fIntervals+1U, -fXmax, fXmax);
            double cdf = m1.cdf(-fXmax);
            for (unsigned i=0; i<fIntervals; ++i)
            {
                const double upper = grid.coordinate(i+1U);
                const double uCdf = m1.cdf(upper);
                d1[i] = uCdf - cdf;
                cdf = uCdf;
            }
        }

        for (unsigned iconv = 0; iconv < nIntervals_; ++iconv)
        {
            long double sum = 0.0L;
            const unsigned start = iconv + nIntervals_ - 1U;
            for (unsigned i = 0; i < nIntervals_; ++i)
                sum += d2[i]*d1[start - i];
            convolution_[iconv] = sum/h;
        }

        if (i_normalize)
            normalize();
    }

    double DiscretizedConvolution::operator()(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        unsigned i = (x - xmin_)/intervalWidth();
        if (i >= nIntervals_)
            i = nIntervals_ - 1U;
        return convolution_[i];
    }

    void DiscretizedConvolution::normalize()
    {
        // Renormalize even if normalized already.
        // Could help with round-offs.
        const double integ = densityIntegral();
        for (unsigned i=0; i<nIntervals_; ++i)
            convolution_[i] /= integ;
        normalized_ = true;
    }

    double DiscretizedConvolution::densityIntegral() const
    {
        return intervalWidth()*std::accumulate(
            convolution_.begin(), convolution_.end(), 0.0L);
    }

    double DiscretizedConvolution::coordinateAt(const unsigned i) const
    {
        if (i >= nIntervals_) throw std::invalid_argument(
            "In ase::DiscretizedConvolution::coordinateAt: "
            "index out of range");
        return xmin_ + (i + 0.5)*intervalWidth();
    }

    TabulatedDensity1D
    DiscretizedConvolution::constructTabulatedDensity() const
    {
        const double hover2 = intervalWidth()/2.0;
        return TabulatedDensity1D(xmin_+hover2, xmax_-hover2, convolution_);
    }

    InterpolatedDensity1D
    DiscretizedConvolution::constructInterpolatedDensity() const
    {
        const double hover2 = intervalWidth()/2.0;
        return InterpolatedDensity1D(xmin_+hover2, xmax_-hover2, convolution_);
    }
}
