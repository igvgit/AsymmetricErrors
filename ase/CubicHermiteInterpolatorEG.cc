#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>

#include "ase/mathUtils.hh"
#include "ase/CubicHermiteInterpolatorEG.hh"
#include "ase/PosteriorMomentFunctor.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace ase {
    CubicHermiteInterpolatorEG::CubicHermiteInterpolatorEG(
        const double minParam, const double maxParam,
        const std::vector<double>& values)
        : grid_(values.size(), minParam, maxParam),
          values_(values),
          derivatives_(values.size())
    {
        const unsigned npt = grid_.nCoords();
        const unsigned nptM1 = npt - 1U;
        const double h = grid_.intervalWidth();
        const double twoh = 2.0*h;
        for (unsigned i=1U; i<nptM1; ++i)
            derivatives_[i] = (values_[i+1U] - values_[i-1U])/twoh;

        if (npt > 2U)
        {
            // The following should give a quadratic fit at the boundary intervals
            derivatives_[0] = -((-2.0*(values_[1] - values_[0]) + derivatives_[1]*h)/h);
            derivatives_[nptM1] = -((2.0*(values_[nptM1-1U] - values_[nptM1]) + derivatives_[nptM1-1U]*h)/h);
        }
        else
        {
            derivatives_[0] = (values_[1] - values_[0])/h;
            derivatives_[nptM1] = (values_[nptM1] - values_[nptM1-1U])/h;
        }

        findMaximum();
        findLocation();
    }

    CubicHermiteInterpolatorEG::CubicHermiteInterpolatorEG(
        const double minParam, const double maxParam,
        const std::vector<double>& values,
        const std::vector<double>& derivs)
        : grid_(values.size(), minParam, maxParam),
          values_(values),
          derivatives_(derivs)
    {
        const unsigned npt = grid_.nCoords();
        if (npt != derivatives_.size()) throw std::invalid_argument(
            "In ase::CubicHermiteInterpolatorEG constructor: "
            "inconsistent sizes of input vectors");
        findMaximum();
        findLocation();
    }

    CubicHermiteInterpolatorEG::CubicHermiteInterpolatorEG(
        const double minParam, const double maxParam,
        const unsigned nScanPoints, const AbsLogLikelihoodCurve& curve)
        : grid_(nScanPoints, minParam, maxParam),
          values_(nScanPoints),
          derivatives_(nScanPoints)
    {
        for (unsigned i=0; i<nScanPoints; ++i)
        {
            const double x = grid_.coordinate(i);
            values_[i] = curve(x);
            derivatives_[i] = curve.derivative(x);
        }
        findMaximum();
        findLocation();
    }

    AbsLogLikelihoodCurve& CubicHermiteInterpolatorEG::operator*=(const double c)
    {
        const unsigned sz = values_.size();
        for (unsigned i=0; i<sz; ++i)
        {
            values_[i] *= c;
            derivatives_[i] *= c;
        }
        if (c >= 0.0)
            logliMax_ *= c;
        else
            findMaximum();
        return *this;
    }

    double CubicHermiteInterpolatorEG::operator()(const double x) const
    {
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double onemt = cellPair.second;
        const double t = 1.0 - onemt;
        const double h00 = onemt*onemt*(1.0 + 2.0*t);
        const double h10 = onemt*onemt*t;
        const double h01 = t*t*(3.0 - 2.0*t);
        const double h11 = t*t*onemt;
        const double h = grid_.intervalWidth();
        return h00*values_[cell] + h10*h*derivatives_[cell] +
               h01*values_[cellp1] - h11*h*derivatives_[cellp1];
    }

    double CubicHermiteInterpolatorEG::derivative(const double x) const
    {
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double onemt = cellPair.second;
        const double t = 1.0 - onemt;
        const double h = grid_.intervalWidth();
        const double d0 = derivatives_[cell];
        const double dsum = d0 + derivatives_[cellp1];
        return 6.0*(values_[cellp1]-values_[cell])*onemt*t/h +
               t*(3.0*t*dsum - 2.0*(dsum + d0)) + d0;
    }

    double CubicHermiteInterpolatorEG::secondDerivative(const double x, double) const
    {
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double onemt = cellPair.second;
        const double t = 1.0 - onemt;
        const double h = grid_.intervalWidth();
        const double d0 = derivatives_[cell];
        const double dsum = d0 + derivatives_[cellp1];
        return (6.0*dsum*t - 2.0*(d0 + dsum) +
                (6.0*(values_[cellp1]-values_[cell])*(1.0 - 2.0*t))/h)/h;
    }

    void CubicHermiteInterpolatorEG::findMaximum()
    {
        const unsigned npt = grid_.nCoords();
        assert(npt == values_.size());
        assert(npt == derivatives_.size());
        const unsigned imax = std::distance(values_.begin(), std::max_element(values_.begin(), values_.end()));
        const double h = grid_.intervalWidth();
        if (imax == 0U)
        {
            const std::pair<double,double>& rmax = cubicMaximum01(
                values_[imax], h*derivatives_[imax],
                values_[imax+1U], h*derivatives_[imax+1U]);
            argmax_ = grid_.coordinate(imax) + rmax.first*h;
        }
        else if (imax + 1U == npt)
        {
            const std::pair<double,double>& lmax = cubicMaximum01(
                values_[imax-1U], h*derivatives_[imax-1U],
                values_[imax], h*derivatives_[imax]);
            argmax_ = grid_.coordinate(imax-1U) + lmax.first*h;
        }
        else
        {
            const std::pair<double,double>& lmax = cubicMaximum01(
                values_[imax-1U], h*derivatives_[imax-1U],
                values_[imax], h*derivatives_[imax]);
            const std::pair<double,double>& rmax = cubicMaximum01(
                values_[imax], h*derivatives_[imax],
                values_[imax+1U], h*derivatives_[imax+1U]);
            if (lmax.second > rmax.second)
                argmax_ = grid_.coordinate(imax-1U) + lmax.first*h;
            else
                argmax_ = grid_.coordinate(imax) + rmax.first*h;
        }
        if (argmax_ < grid_.min())
            argmax_ = grid_.min();
        else if (argmax_ > grid_.max())
            argmax_ = grid_.max();
        logliMax_ = (*this)(argmax_);
    }

    std::pair<double,double> CubicHermiteInterpolatorEG::findMinimum() const
    {
        const unsigned npt = grid_.nCoords();
        assert(npt == values_.size());
        assert(npt == derivatives_.size());
        const unsigned imax = std::distance(values_.begin(), std::min_element(values_.begin(), values_.end()));
        const double h = grid_.intervalWidth();
        double argmin;
        if (imax == 0U)
        {
            const std::pair<double,double>& rmin = cubicMinimum01(
                values_[imax], h*derivatives_[imax],
                values_[imax+1U], h*derivatives_[imax+1U]);
            argmin = grid_.coordinate(imax) + rmin.first*h;
        }
        else if (imax + 1U == npt)
        {
            const std::pair<double,double>& lmin = cubicMinimum01(
                values_[imax-1U], h*derivatives_[imax-1U],
                values_[imax], h*derivatives_[imax]);
            argmin = grid_.coordinate(imax-1U) + lmin.first*h;
        }
        else
        {
            const std::pair<double,double>& lmin = cubicMinimum01(
                values_[imax-1U], h*derivatives_[imax-1U],
                values_[imax], h*derivatives_[imax]);
            const std::pair<double,double>& rmin = cubicMinimum01(
                values_[imax], h*derivatives_[imax],
                values_[imax+1U], h*derivatives_[imax+1U]);
            if (lmin.second < rmin.second)
                argmin = grid_.coordinate(imax-1U) + lmin.first*h;
            else
                argmin = grid_.coordinate(imax) + rmin.first*h;
        }
        if (argmin < grid_.min())
            argmin = grid_.min();
        else if (argmin > grid_.max())
            argmin = grid_.max();
        return std::pair<double,double>(argmin, (*this)(argmin));
    }

    void CubicHermiteInterpolatorEG::findLocation()
    {
        const double h = grid_.intervalWidth();
        const double smallDelta = h*sqrt(std::numeric_limits<double>::epsilon());
        if (argmax_ > grid_.min()+smallDelta && argmax_ < grid_.max()-smallDelta)
            // argmax_ is likely to be a real local maximum
            location_  = argmax_;
        else
        {
            const double argmin = findMinimum().first;
            if (argmin > grid_.min()+smallDelta && argmin < grid_.max()-smallDelta)
                location_  = argmin;
            else
            {
                // Choose the point with the largest absolute second derivative
                const unsigned npt = grid_.nCoords();
                const unsigned nptM1 = npt - 1U;
                unsigned imax = 0U;
                double d2max = std::abs(derivatives_[1] - derivatives_[0]);
                for (unsigned i=1; i<nptM1; ++i)
                {
                    const double d2 = std::abs(derivatives_[i+1U] - derivatives_[i]);
                    if (d2 > d2max)
                    {
                        imax = i;
                        d2max = d2;
                    }
                }
                location_ = (grid_.coordinate(imax) + grid_.coordinate(imax+1U))/2.0;
            }
        }
    }

    double CubicHermiteInterpolatorEG::unnormalizedMoment(
        const double p0, const unsigned n, double /* maxDeltaLogli */) const
    {
        const GaussLegendreQuadrature glq(8U);
        const PosteriorMomentFunctor momFcn(*this, p0, n);
        return glq.integrate(momFcn, grid_.min(), grid_.max(), grid_.nIntervals());
    }
}
