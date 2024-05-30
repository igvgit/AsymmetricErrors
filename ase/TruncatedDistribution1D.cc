#include <stdexcept>
#include <algorithm>

#include "ase/TruncatedDistribution1D.hh"

namespace ase {
    TruncatedDistribution1D::TruncatedDistribution1D(
        const AbsDistributionModel1D& distro,
        const double i_xmin, const double i_xmax)
        : distro_(distro), xmin_(i_xmin), xmax_(i_xmax)
    {
        if (xmin_ > xmax_)
            std::swap(xmin_, xmax_);
        if (xmin_ == xmax_) throw std::invalid_argument(
            "In ase::TruncatedDistribution1D constructor: "
            "can not use the same value for the support limits");
        xmin_ = std::max(xmin_, distro_.quantile(0.0));
        xmax_ = std::min(xmax_, distro_.quantile(1.0));
        cdfmin_ = distro_.cdf(xmin_);
        cdfmax_ = distro_.cdf(xmax_);
        if (cdfmax_ == cdfmin_) throw std::invalid_argument(
            "In ase::TruncatedDistribution1D constructor: "
            "can not use the same value for the minimum and maximum cdf");
        exmin_ = distro_.exceedance(xmax_);
    }

    double TruncatedDistribution1D::density(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else
            return distro_.density(x)/(cdfmax_ - cdfmin_);
    }

    double TruncatedDistribution1D::densityDerivative(const double x) const
    {
        if (x < xmin_ || x > xmax_)
            return 0.0;
        else
            return distro_.densityDerivative(x)/(cdfmax_ - cdfmin_);
    }

    double TruncatedDistribution1D::cdf(const double x) const
    {
        if (x <= xmin_)
            return 0.0;
        else if (x >= xmax_)
            return 1.0;
        else
            return (distro_.cdf(x) - cdfmin_)/(cdfmax_ - cdfmin_);
    }

    double TruncatedDistribution1D::exceedance(const double x) const
    {
        if (x <= xmin_)
            return 1.0;
        else if (x >= xmax_)
            return 0.0;
        else
            return (distro_.exceedance(x) - exmin_)/(cdfmax_ - cdfmin_);
    }

    double TruncatedDistribution1D::quantile(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::TruncatedDistribution1D::quantile: "
            "cdf argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return xmin_;
        else if (r1 == 1.0)
            return xmax_;
        else
            return distro_.quantile(r1*(cdfmax_ - cdfmin_) + cdfmin_);
    }

    double TruncatedDistribution1D::invExceedance(const double r1) const
    {
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::TruncatedDistribution1D::quantile: "
            "exceedance argument outside of [0, 1] interval");
        if (r1 == 0.0)
            return xmax_;
        else if (r1 == 1.0)
            return xmin_;
        else
            return distro_.invExceedance(r1*(cdfmax_ - cdfmin_) + exmin_);
    }

    bool TruncatedDistribution1D::isDensityContinuous() const
    {
        return distro_.isDensityContinuous();
    }

    bool TruncatedDistribution1D::isNonNegative() const
    {
        return distro_.isNonNegative();
    }

    bool TruncatedDistribution1D::isUnimodal() const
    {
        return distro_.isUnimodal();
    }

    double TruncatedDistribution1D::mode() const
    {
        const double oldMode = distro_.mode();
        if (xmin_ <= oldMode && oldMode <= xmax_)
            return oldMode;
        else
        {
            if (distro_.isUnimodal())
            {
                if (distro_.density(xmin_) >= distro_.density(xmax_))
                    return xmin_;
                else
                    return xmax_;
            }
            else
            {
                throw std::runtime_error("In ase::TruncatedDistribution1D::mode: "
                                         "general mode search is not implemented");
                return 0.0;
            }
        }
    }

    double TruncatedDistribution1D::descentDelta(
        bool /* isToTheRight */, double /* deltaLnL */) const
    {
        throw std::runtime_error("In ase::TruncatedDistribution1D::descentDelta: "
                                 "this method is not implemented");
        return 0.0;
    }

    double TruncatedDistribution1D::cumulant(unsigned /* n */) const
    {
        throw std::runtime_error("In ase::TruncatedDistribution1D::cumulant: "
                                 "this method is not implemented");
        return 0.0;
    }
}
