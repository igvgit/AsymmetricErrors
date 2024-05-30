#include <cmath>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <functional>
#include <stdexcept>

#include "ase/TabulatedDensity1D.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/GaussLegendreQuadrature.hh"
#include "ase/mathUtils.hh"

namespace ase {
    void TabulatedDensity1D::calculateCumulants() const
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

    double TabulatedDensity1D::unscaledCumulant(const unsigned n) const
    {
        if (n > 4U) throw std::invalid_argument(
            "In ase::TabulatedDensity1D::unscaledCumulant: "
            "only four leading cumulants are implemented");
        if (!cumulantsCalculated_)
            calculateCumulants();
        return cumulants_[n];
    }

    double TabulatedDensity1D::unscaledDensity(const double x) const
    {
        if (x < grid_.min() || x > grid_.max())
            return 0.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const double w = cellPair.second;
        return w*values_[cell] + (1.0 - w)*values_[cell + 1U];
    }

    double TabulatedDensity1D::unscaledDensityDerivative(const double x) const
    {
        if (x < grid_.min() || x > grid_.max())
            return 0.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const double h = grid_.intervalWidth();
        return (values_[cell + 1U] - values_[cell])/h;
    }

    double TabulatedDensity1D::unscaledCdf(const double x) const
    {
        if (x <= grid_.min())
            return 0.0;
        if (x >= grid_.max())
            return 1.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const double w = cellPair.second;
        const double onemw = 1.0 - w;
        const double h = grid_.intervalWidth();
        return cdfValues_[cell] + ((1.0 + w)*values_[cell] + onemw*values_[cell + 1U])/2.0*h*onemw;
    }

    double TabulatedDensity1D::unscaledExceedance(const double x) const
    {
        if (x <= grid_.min())
            return 1.0;
        if (x >= grid_.max())
            return 0.0;
        const std::pair<unsigned,double>& cellPair = grid_.getInterval(x);
        const unsigned cell = cellPair.first;
        const unsigned cellp1 = cell + 1U;
        const double w = cellPair.second;
        const double h = grid_.intervalWidth();
        return excValues_[cellp1] + (w*values_[cell] + (2.0 - w)*values_[cellp1])/2.0*h*w;
    }

    double TabulatedDensity1D::unscaledQuantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::TabulatedDensity1D::unscaledQuantile: "
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

        /*
        double q;
        const bool status = findRootUsingBisections(
            UnscaledCdfFunctor1D(*this), r1, xmin, grid_.coordinate(uabove), tol, &q);
        assert(status);
        return q;
        */
        const double h = grid_.intervalWidth();
        const double tmp = (r1 - cdfValues_[ubelow])*2.0/h;
        double w;
        if (values_[ubelow] == values_[uabove])
        {
            assert(values_[ubelow]);
            w = 1.0 - tmp/2.0/values_[ubelow];
        }
        else
        {
            const double del = values_[uabove] - values_[ubelow];
            const double b = -2.0*values_[uabove]/del;
            const double c = (values_[uabove] + values_[ubelow] - tmp)/del;
            double soln[2];
            const unsigned nRoots = solveQuadratic(b, c, &soln[0], &soln[1]);
            assert(nRoots == 2U);
            if (std::abs(soln[0] - 0.5) < std::abs(soln[1] - 0.5))
                w = soln[0];
            else
                w = soln[1];
            if (w < 0.0)
                w = 0.0;
            if (w > 1.0)
                w = 1.0;
        }
        return xmin + h*(1.0 - w);
    }

    double TabulatedDensity1D::unscaledInvExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::TabulatedDensity1D::unscaledInvExceedance: "
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

        /*
        double q;
        const bool status = findRootUsingBisections(
            UnscaledExceedanceFunctor1D(*this), r1, xmin, grid_.coordinate(uabove), tol, &q);
        assert(status);
        return q;
        */
        const double h = grid_.intervalWidth();
        const double tmp = (r1 - excValues_[uabove])*2.0/h;
        double w;
        if (values_[ubelow] == values_[uabove])
        {
            assert(values_[ubelow]);
            w = tmp/2.0/values_[ubelow];
        }
        else
        {
            const double del = values_[ubelow] - values_[uabove];
            const double b = 2.0*values_[uabove]/del;
            const double c = -tmp/del;
            double soln[2];
            const unsigned nRoots = solveQuadratic(b, c, &soln[0], &soln[1]);
            assert(nRoots == 2U);
            if (std::abs(soln[0] - 0.5) < std::abs(soln[1] - 0.5))
                w = soln[0];
            else
                w = soln[1];
            if (w < 0.0)
                w = 0.0;
            if (w > 1.0)
                w = 1.0;
        }
        return xmin + h*(1.0 - w);
    }

    void TabulatedDensity1D::normalize()
    {
        // We only want to call this function once
        assert(cdfValues_.empty());

        // Make sure that the density does not become negative anywhere
        const unsigned npt = grid_.nCoords();
        if (npt < 2U) throw std::invalid_argument(
            "In ase::TabulatedDensity1D::normalize: "
            "insufficient number of interpolation points");

        for (unsigned i=0; i<npt; ++i)
            if (values_[i] < 0.0) throw std::invalid_argument(
                "In ase::TabulatedDensity1D::normalize: "
                "negative input density value encountered");

        // Integrate the density on the intervals and build cdf/exceedance tables
        const double halfh = grid_.intervalWidth()/2.0;
        const unsigned nptM1 = npt - 1U;
        cdfValues_.reserve(npt);
        cdfValues_.push_back(0.0);
        excValues_.resize(npt);
        excValues_[nptM1] = 0.0;

        long double sum = 0.0L;
        for (unsigned i=0; i<nptM1; ++i)
        {
            sum += (values_[i] + values_[i+1U])*halfh;
            cdfValues_.push_back(sum);
        }
        const double norm = sum;
        if (norm <= 0.0) throw std::invalid_argument(
            "In ase::TabulatedDensity1D::normalize: "
            "density integral is not positive");
        for (unsigned i=0; i<npt; ++i)
        {
            values_[i] /= norm;
            cdfValues_[i] /= norm;
        }

        sum = 0.0L;
        for (long i=nptM1-1; i>=0; --i)
        {
            sum += (values_[i] + values_[i+1])*halfh;
            excValues_[i] = sum;
        }
        const double excNorm = sum;
        assert(excNorm > 0.0);
        for (unsigned i=0; i<npt; ++i)
            excValues_[i] /= excNorm;
    }

    double TabulatedDensity1D::calculateMoment(
        const double mu, const unsigned power) const
    {
        const unsigned nInteg = GaussLegendreQuadrature::minimalExactRule(power + 1U);
        const GaussLegendreQuadrature glq(nInteg);
        const UnscaledMomentFunctor1D fcn(*this, mu, power);
        return glq.integrate(fcn, grid_.min(), grid_.max(), grid_.nCoords() - 1U);
    }

    double TabulatedDensity1D::unscaledEntropy() const
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

    double TabulatedDensity1D::unscaledMode() const
    {
        const unsigned long i = std::max_element(values_.begin(), values_.end())
            - values_.begin();
        return grid_.coordinate(i);
    }

    bool TabulatedDensity1D::determineUnimodality() const
    {
        if (!unimodalityDetermined_)
        {
            const unsigned npt = grid_.nCoords();
            assert(npt >= 2U);
            unimodal_ = true;
            if (npt > 2U)
            {
                double leftMaxValue = values_[0];
                const unsigned nptM1 = npt - 1U;
                for (unsigned i=1; i<nptM1 && unimodal_; ++i)
                {
                    const double v = values_[i];
                    if (v <= values_[i-1U] && v <= values_[i+1U])
                    {
                        bool haveRightHigher = false;
                        for (unsigned k=i+1U; k<npt; ++k)
                            if (values_[k] > v)
                            {
                                haveRightHigher = true;
                                break;
                            }
                        if (haveRightHigher && v < leftMaxValue)
                            unimodal_ = false;
                    }
                    if (v > leftMaxValue)
                        leftMaxValue = v;
                }
            }

            unimodalityDetermined_ = true;
        }
        return unimodal_;
    }
}
