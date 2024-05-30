#include <cassert>
#include <limits>
#include <sstream>
#include <algorithm>
#include <functional>

#include "ase/InterpolatedDensity1D.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/mathUtils.hh"

namespace ase {
    InterpolatedDensity1D::InterpolatedDensity1D(
        const double unscaledXmin, const double unscaledXmax,
        const std::vector<double>& unscaledDensityValues)
        : AbsLocationScaleFamily(0.0, 1.0),
          grid_(unscaledDensityValues.size(), unscaledXmin, unscaledXmax),
          values_(unscaledDensityValues),
          derivatives_(unscaledDensityValues.size()),
          cumulantsCalculated_(false),
          entropyCalculated_(false),
          unimodalityDetermined_(false)
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
        normalize();
    }

    InterpolatedDensity1D::InterpolatedDensity1D(
        const double unscaledXmin, const double unscaledXmax,
        const std::vector<double>& unscaledDensityValues,
        const std::vector<double>& unscaledDensityDerivs)
        : AbsLocationScaleFamily(0.0, 1.0),
          grid_(unscaledDensityValues.size(), unscaledXmin, unscaledXmax),
          values_(unscaledDensityValues),
          derivatives_(unscaledDensityDerivs),
          cumulantsCalculated_(false),
          entropyCalculated_(false),
          unimodalityDetermined_(false)
    {
        const unsigned npt = grid_.nCoords();
        if (npt != derivatives_.size()) throw std::invalid_argument(
            "In ase::InterpolatedDensity1D constructor: "
            "inconsistent sizes of input vectors");
        normalize();
    }

    void InterpolatedDensity1D::calculateCumulants() const
    {
        cumulants_[0] = 1.0;
        const double mu = calculateMoment(0.0, 1U);
        cumulants_[1] = mu;
        const double var = calculateMoment(mu, 2U);
        cumulants_[2] = var;
        cumulants_[3] = calculateMoment(mu, 3U);
        cumulants_[4] = calculateMoment(mu, 4U) - 3.0*var*var;
        cumulantsCalculated_ = true;
    }

    double InterpolatedDensity1D::unscaledCumulant(const unsigned n) const
    {
        if (n > 4U) throw std::invalid_argument(
            "In ase::InterpolatedDensity1D::unscaledCumulant: "
            "only four leading cumulants are implemented");
        if (!cumulantsCalculated_)
            calculateCumulants();
        return cumulants_[n];
    }

    double InterpolatedDensity1D::unscaledDensity(const double x) const
    {
        if (x < grid_.min() || x > grid_.max())
            return 0.0;
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

    double InterpolatedDensity1D::unscaledDensityDerivative(const double x) const
    {
        if (x < grid_.min() || x > grid_.max())
            return 0.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double onemt = cellPair.second;
        const double t = 1.0 - onemt;
        const double dh00 = 6.0*t*(t - 1.0);
        const double dh10 = onemt*(1.0 - 3.0*t);
        const double dh01 = -dh00;
        const double dh11 = (2.0 - 3.0*t)*t;
        const double h = grid_.intervalWidth();
        return dh00*values_[cell]/h + dh10*derivatives_[cell] +
               dh01*values_[cellp1]/h - dh11*derivatives_[cellp1];
    }

    double InterpolatedDensity1D::unscaledCdf(const double x) const
    {
        if (x <= grid_.min())
            return 0.0;
        if (x >= grid_.max())
            return 1.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double onemt = cellPair.second;
        const double t = 1.0 - onemt;
        const double tsq = t*t;
        const double t3 = tsq*t;
        const double h00i = t*(1.0 + tsq*(t/2.0 - 1.0));
        const double h10i = tsq*(0.5 + t*(t/4.0 - 2.0/3.0));
        const double h01i = t3*(1.0 - t/2.0);
        const double h11i = t3*(1.0/3.0 - t/4.0);
        const double h = grid_.intervalWidth();
        return h*(h00i*values_[cell] + h10i*h*derivatives_[cell] +
                  h01i*values_[cellp1] - h11i*h*derivatives_[cellp1]) +
               cdfValues_[cell];
    }

    double InterpolatedDensity1D::unscaledExceedance(const double x) const
    {
        if (x <= grid_.min())
            return 1.0;
        if (x >= grid_.max())
            return 0.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double w = cellPair.second;
        const double t = 1.0 - w;
        const double w3 = w*w*w;
        const double t3 = t*t*t;
        const double h00c = w3*(1.0 - w/2.0);
        const double h10c = w3*(1.0/3.0 - w/4.0);
        const double h01c = 0.5 + t3*(t/2.0 - 1.0);
        const double h11c = 1.0/12.0 + t3*(t/4.0 - 1.0/3.0);
        const double h = grid_.intervalWidth();
        return h*(h00c*values_[cell] + h10c*h*derivatives_[cell] +
                  h01c*values_[cellp1] - h11c*h*derivatives_[cellp1]) +
               excValues_[cellp1];
    }

    double InterpolatedDensity1D::unscaledQuantile(const double r1) const
    {
        static const double tol = 2.0*std::numeric_limits<double>::epsilon();

        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::InterpolatedDensity1D::unscaledQuantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return grid_.min();
        if (r1 == 1.0)
            return grid_.max();

        const std::size_t iabove =
            std::upper_bound(cdfValues_.begin(), cdfValues_.end(), r1) - cdfValues_.begin();
        assert(iabove > 0);
        const unsigned uabove = static_cast<unsigned>(iabove);
        assert(uabove < grid_.nCoords());
        const unsigned ubelow = uabove - 1U;
        const double xmin = grid_.coordinate(ubelow);
        if (cdfValues_[ubelow] == r1)
            return xmin;

        double q;
        const bool status = findRootUsingBisections(
            UnscaledCdfFunctor1D(*this), r1, xmin, grid_.coordinate(uabove), tol, &q);
        assert(status);
        return q;
    }

    double InterpolatedDensity1D::unscaledInvExceedance(const double r1) const
    {
        static const double tol = 2.0*std::numeric_limits<double>::epsilon();

        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::InterpolatedDensity1D::unscaledInvExceedance: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 1.0)
            return grid_.min();
        if (r1 == 0.0)
            return grid_.max();

        const std::greater<double> comp;
        const std::size_t iabove =
            std::upper_bound(excValues_.begin(), excValues_.end(), r1, comp) - excValues_.begin();
        assert(iabove > 0);
        const unsigned uabove = static_cast<unsigned>(iabove);
        assert(uabove < grid_.nCoords());
        const unsigned ubelow = uabove - 1U;
        const double xmin = grid_.coordinate(ubelow);
        if (excValues_[ubelow] == r1)
            return xmin;

        double q;
        const bool status = findRootUsingBisections(
            UnscaledExceedanceFunctor1D(*this), r1, xmin, grid_.coordinate(uabove), tol, &q);
        assert(status);
        return q;
    }

    void InterpolatedDensity1D::normalize()
    {
        // We only want to call this function once
        assert(cdfValues_.empty());

        // Make sure that the density does not become negative anywhere
        const unsigned npt = grid_.nCoords();
        if (npt < 2U) throw std::invalid_argument(
            "In ase::InterpolatedDensity1D::normalize: "
            "insufficient number of interpolation points");

        for (unsigned i=0; i<npt; ++i)
            if (values_[i] < 0.0) throw std::invalid_argument(
                "In ase::InterpolatedDensity1D::normalize: "
                "negative input density value encountered");

        const double h = grid_.intervalWidth();
        const unsigned nptM1 = npt - 1U;
        for (unsigned i=0; i<nptM1; ++i)
        {
            const double minValue = cubicMinimum01(
                values_[i], h*derivatives_[i],
                values_[i+1U], h*derivatives_[i+1U]).second;
            if (minValue < 0.0)
            {
                std::ostringstream os;
                os << "In ase::InterpolatedDensity1D::normalize: "
                   << "input density values and derivatives make "
                   << "the density negative on the interval " << i;
                throw std::invalid_argument(os.str());
            }
        }

        // Integrate the density on the intervals and build cdf/exceedance tables
        const unsigned nInteg = GaussLegendreQuadrature::minimalExactRule(3U);
        const GaussLegendreQuadrature glq(nInteg);
        const UnscaledDensityFunctor1D fcn(*this);
        cdfValues_.reserve(npt);
        cdfValues_.push_back(0.0);
        excValues_.resize(npt);
        excValues_[nptM1] = 0.0;

        std::vector<long double> integrals(nptM1);
        for (unsigned i=0; i<nptM1; ++i)
        {
            const double xmin = grid_.coordinate(i);
            const double xmax = grid_.coordinate(i+1U);
            const long double integ = glq.integrate(fcn, xmin, xmax);
            assert(integ >= 0.0);
            integrals[i] = integ;
        }

        long double sum = 0.0L;
        for (unsigned i=0; i<nptM1; ++i)
        {
            sum += integrals[i];
            cdfValues_.push_back(sum);
        }
        const double norm = sum;
        if (norm <= 0.0) throw std::invalid_argument(
            "In ase::InterpolatedDensity1D::normalize: "
            "density integral is not positive");
        for (unsigned i=0; i<npt; ++i)
        {
            values_[i] /= norm;
            derivatives_[i] /= norm;
            cdfValues_[i] /= norm;
        }

        sum = 0.0L;
        for (long i=nptM1-1; i>=0; --i)
        {
            sum += integrals[i];
            excValues_[i] = sum;
        }
        const double excNorm = sum;
        assert(excNorm > 0.0);
        for (unsigned i=0; i<npt; ++i)
            excValues_[i] /= excNorm;
    }

    double InterpolatedDensity1D::calculateMoment(
        const double mu, const unsigned power) const
    {
        const unsigned nInteg = GaussLegendreQuadrature::minimalExactRule(power + 3U);
        const GaussLegendreQuadrature glq(nInteg);
        const UnscaledMomentFunctor1D fcn(*this, mu, power);
        return glq.integrate(fcn, grid_.min(), grid_.max(), grid_.nCoords() - 1U);
    }

    double InterpolatedDensity1D::unscaledEntropy() const
    {
        if (!entropyCalculated_)
        {
            const GaussLegendreQuadrature glq(100);
            const UnscaledEntropyFunctor1D fcn(*this);
            entropy_ = glq.integrate(fcn, grid_.min(), grid_.max(), grid_.nCoords() - 1U);
            entropyCalculated_ = true;
        }
        return entropy_;
    }

    bool InterpolatedDensity1D::determineUnimodality() const
    {
        if (!unimodalityDetermined_)
        {
            // The curve is unimodal if there are no minima on the
            // interval (grid_.min(), grid_.max()), i.e., no such
            // points at which the derivative is 0 and the second
            // derivative is positive. Note that such minima are
            // allowed at the boundaries grid_.min() and grid_.max().
            const unsigned npt = grid_.nCoords();
            const unsigned nptM1 = npt - 1U;
            const unsigned nptM2 = npt - 2U;
            const double h = grid_.intervalWidth();

            unimodal_ = true;
            for (unsigned cell=0; cell<nptM1 && unimodal_; ++cell)
            {
                const unsigned cellp1 = cell + 1U;
                const double v0 = values_[cell];
                const double v1 = values_[cellp1];
                const double d0 = derivatives_[cell];
                const double d1 = derivatives_[cellp1];

                // Derivative is quadratic: a t^2 + b t + c
                const double tmp = 6*(v1 - v0)/h;
                const double a = 3*(d0 + d1) - tmp;
                const double b = tmp - 2*d1 - 4*d0;
                const double c = d0;

                if (a)
                {
                    double t[2];
                    const unsigned nRoots = solveQuadratic(b/a, c/a, &t[0], &t[1]);
                    for (unsigned irt=0; irt<nRoots; ++irt)
                    {
                        const double rt = t[irt];
                        if (rt >= 0.0 && rt <= 1.0)
                        {
                            if (rt == 0.0 && cell == 0)
                                continue;
                            if (rt == 1.0 && cell == nptM2)
                                continue;
                            const double secondDeriv = 2.0*a*rt + b;
                            if (secondDeriv > 0.0)
                                unimodal_ = false;
                        }
                    }
                }
                else if (b > 0.0)
                {
                    const double rt = -c/b;
                    if (rt >= 0.0 && rt <= 1.0)
                    {
                        if (rt == 0.0 && cell == 0)
                            continue;
                        if (rt == 1.0 && cell == nptM2)
                            continue;
                        unimodal_ = false;
                    }
                }
            }

            unimodalityDetermined_ = true;
        }
        return unimodal_;
    }
}
