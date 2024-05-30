#include <cmath>
#include <cfloat>
#include <vector>
#include <numeric>

#include "ase/MixtureModel1D.hh"
#include "ase/statUtils.hh"

namespace ase {
    const double MixtureModel1D::tol_ = 2.0*DBL_EPSILON;
    const double MixtureModel1D::sqrtol_ = std::sqrt(MixtureModel1D::tol_);

    MixtureModel1D& MixtureModel1D::add(
        const AbsDistributionModel1D& distro, const double weight)
    {
        if (!distro.isNonNegative()) throw std::invalid_argument(
            "In ase::MixtureModel1D::add: can't mix distributions allowing "
            "negative densities");
        if (weight < 0.0) throw std::invalid_argument(
            "In ase::MixtureModel1D::add: can't use negative weights");
        if (weight > 0.0)
        {
            entries_.emplace_back(distro);
            weights_.push_back(weight);
            isNormalized_ = false;
        }
        return *this;
    }

    void MixtureModel1D::normalize()
    {
        const unsigned nentries = entries_.size();
        if (!nentries) throw std::runtime_error(
            "In ase::MixtureModel1D::normalize: no components in the mixture");
        weightCdf_.clear();
        weightCdf_.reserve(nentries);
        const long double wnorm = std::accumulate(
            weights_.begin(), weights_.end(), 0.0L);
        assert(wnorm > 0.0L);
        wsum_ = 0.0L;
        for (unsigned i=0; i<nentries; ++i)
        {
            weightCdf_.push_back(static_cast<double>(wsum_/wnorm));
            wsum_ += weights_[i];
        }
        isNormalized_ = true;
    }

    double MixtureModel1D::getWeight(const unsigned n) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        return weights_.at(n)/wsum_;
    }

    double MixtureModel1D::density(const double x) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        long double sum = 0.0L;
        for (unsigned i=0; i<nentries; ++i)
            sum += weights_[i]*entries_[i].density(x);
        return sum/wsum_;
    }

    double MixtureModel1D::densityDerivative(const double x) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        long double sum = 0.0L;
        for (unsigned i=0; i<nentries; ++i)
            sum += weights_[i]*entries_[i].densityDerivative(x);
        return sum/wsum_;
    }

    bool MixtureModel1D::isDensityContinuous() const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        for (unsigned i=0; i<nentries; ++i)
            if (!entries_[i].isDensityContinuous())
                return false;
        return true;
    }

    double MixtureModel1D::cdf(const double x) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        long double sum = 0.0L;
        for (unsigned i=0; i<nentries; ++i)
            sum += weights_[i]*entries_[i].cdf(x);
        return sum/wsum_;
    }

    double MixtureModel1D::exceedance(const double x) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        long double sum = 0.0L;
        for (unsigned i=0; i<nentries; ++i)
            sum += weights_[i]*entries_[i].exceedance(x);
        return sum/wsum_;
    }

    double MixtureModel1D::quantile(const double r1) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        if (nentries == 1U)
            return entries_[0].quantile(r1);
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::MixtureModel1D::quantile: "
            "cdf argument outside of [0, 1] interval");
        double qmax = -DBL_MAX;
        double qmin = DBL_MAX;
        for (unsigned i=0; i<nentries; ++i)
        {
            const double q = entries_[i].quantile(r1);
            if (q > qmax)
                qmax = q;
            if (q < qmin)
                qmin = q;
        }
        if (qmax == qmin)
            return qmin;
        if (r1 == 1.0)
            return qmax;
        if (r1 == 0.0)
            return qmin;
        const double fmin = cdf(qmin);
        const double fmax = cdf(qmax);
        if (!(fmin < r1 && r1 < fmax)) throw std::runtime_error(
            "In ase::MixtureModel1D::quantile: "
            "algorithm precondition error");
        for (unsigned i=0; i<2000U; ++i)
        {
            const double x = (qmin + qmax)/2.0;
            if (std::abs(qmax - qmin)/(std::max(std::abs(qmin), std::abs(qmax)) + sqrtol_) < tol_)
                return x;
            const double fval = cdf(x);
            if (fval == r1)
                return x;
            else if (fval > r1)
                qmax = x;
            else
                qmin = x;
            if (qmax == qmin)
                return qmin;
        }
        return (qmin + qmax)/2.0;
    }

    double MixtureModel1D::invExceedance(const double r1) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        if (nentries == 1U)
            return entries_[0].invExceedance(r1);
        if (!(r1 >= 0.0 && r1 <= 1.0)) throw std::domain_error(
            "In ase::MixtureModel1D::invExceedance: "
            "exceedance argument outside of [0, 1] interval");
        double qmax = -DBL_MAX;
        double qmin = DBL_MAX;
        for (unsigned i=0; i<nentries; ++i)
        {
            const double q = entries_[i].invExceedance(r1);
            if (q > qmax)
                qmax = q;
            if (q < qmin)
                qmin = q;
        }
        if (qmax == qmin)
            return qmin;
        if (r1 == 1.0)
            return qmin;
        if (r1 == 0.0)
            return qmax;
        const double fmin = exceedance(qmin);
        const double fmax = exceedance(qmax);
        if (!(fmin > r1 && r1 > fmax)) throw std::runtime_error(
            "In ase::MixtureModel1D::invExceedance: "
            "algorithm precondition error");
        for (unsigned i=0; i<2000U; ++i)
        {
            const double x = (qmin + qmax)/2.0;
            if (std::abs(qmax - qmin)/(std::max(std::abs(qmin), std::abs(qmax)) + sqrtol_) < tol_)
                return x;
            const double fval = exceedance(x);
            if (fval == r1)
                return x;
            else if (fval > r1)
                qmin = x;
            else
                qmax = x;
            if (qmax == qmin)
                return qmin;
        }
        return (qmin + qmax)/2.0;
    }

    bool MixtureModel1D::isUnimodal() const
    {
        throw std::runtime_error("In ase::MixtureModel1D::isUnimodal: "
                                 "this method is not implemented");
        return false;
    }

    double MixtureModel1D::mode() const
    {
        throw std::runtime_error("In ase::MixtureModel1D::mode: "
                                 "this method is not implemented");
        return 0.0;
    }

    double MixtureModel1D::descentDelta(bool /* isToTheRight */,
                                        double /* deltaLnL */) const
    {
        throw std::runtime_error("In ase::MixtureModel1D::descentDelta: "
                                 "this method is not implemented");
        return 0.0;
    }

    double MixtureModel1D::random(AbsRNG& gen) const
    {
        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        if (nentries == 1U)
            return entries_[0].random(gen);
        else
        {
            const unsigned bin = quantileBinFromCdf(&weightCdf_[0], nentries, gen());
            return entries_.at(bin).random(gen);
        }
    }

    double MixtureModel1D::cumulant(const unsigned n) const
    {
        if (n > 4U)
        {
            throw std::invalid_argument(
                "In ase::MixtureModel1D::cumulant: "
                "only four leading cumulants are implemented");
            return 0.0;
        }

        if (!isNormalized_)
            (const_cast<MixtureModel1D*>(this))->normalize();
        const unsigned nentries = entries_.size();
        if (nentries == 1U)
            return entries_[0].cumulant(n);

        if (n)
        {
            long double mixMoments[5] = {0.0L, 0.0L, 0.0L, 0.0L, 0.0L};
            long double cumBuf[5];
            for (unsigned ient=0; ient<nentries; ++ient)
            {
                const long double w = weights_[ient];
                for (unsigned i=1; i<=n; ++i)
                    cumBuf[i] = entries_[ient].cumulant(i);
                cumulantsToMoments(cumBuf+1, cumBuf+1, n);
                for (unsigned i=1; i<=n; ++i)
                    mixMoments[i] += w*cumBuf[i];
            }
            for (unsigned i=1; i<=n; ++i)
                mixMoments[i] /= wsum_;
            momentsToCumulants(mixMoments+1, mixMoments+1, n);
            return mixMoments[n];
        }
        else
            return 1.0;
    }
}
