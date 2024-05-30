#include <cfloat>
#include <stdexcept>
#include <algorithm>

#include "ase/Interval.hh"
#include "ase/LikelihoodAccumulator.hh"

namespace ase {
    LikelihoodAccumulator::LikelihoodAccumulator()
        : factor_(1.0), pmin_(DBL_MAX), pmax_(-DBL_MAX), loc_(DBL_MAX),
          stepSz_(-DBL_MAX), maxValue_(-DBL_MAX), maxArg_(-DBL_MAX)
    {
        tagModified();
    }

    void LikelihoodAccumulator::tagModified()
    {
        supportCalculated_ = false;
        locationCalculated_ = false;
        stepCalculated_  = false;
        maximumFound_ = false;
    }

    void LikelihoodAccumulator::accumulate(const AbsLogLikelihoodCurve& r)
    {
        {
            const LikelihoodCurveCopy* ptr = dynamic_cast<const LikelihoodCurveCopy*>(&r);
            if (ptr)
            {
                this->accumulate(ptr->theCopy());
                return;
            }
        }
        {
            const LikelihoodCurveSum* ptr = dynamic_cast<const LikelihoodCurveSum*>(&r);
            if (ptr)
            {
                this->accLikelihoodCurveSum(*ptr);
                return;
            }
        }
        {
            const LikelihoodCurveDifference* ptr = dynamic_cast<const LikelihoodCurveDifference*>(&r);
            if (ptr)
            {
                this->accLikelihoodCurveDifference(*ptr);
                return;
            }
        }
        {
            const LikelihoodAccumulator* ptr = dynamic_cast<const LikelihoodAccumulator*>(&r);
            if (ptr)
            {
                this->accLikelihoodAccumulator(*ptr);
                return;
            }
        }
        accSimpleCurve(r);
    }

    void LikelihoodAccumulator::accLikelihoodCurveSum(
        const LikelihoodCurveSum& r)
    {
        const double f = r.factor();
        if (f == 1.0)
        {
            *this += r.leftOperand().theCopy();
            *this += r.rightOperand().theCopy();
        }
        else
        {
            *this += (f*r.leftOperand()).theCopy();
            *this += (f*r.rightOperand()).theCopy();
        }
    }

    void LikelihoodAccumulator::accLikelihoodCurveDifference(
        const LikelihoodCurveDifference& r)
    {
        const double f = r.factor();
        if (f == 1.0)
            *this += r.leftOperand().theCopy();
        else
            *this += (f*r.leftOperand()).theCopy();
        *this += ((-f)*r.rightOperand()).theCopy();
    }

    void LikelihoodAccumulator::accLikelihoodAccumulator(
        const LikelihoodAccumulator& r)
    {
        const double f = r.factor();
        const unsigned n = r.size();
        if (f == 1.0)
        {
            for (unsigned i=0; i<n; ++i)
                *this += r[i].theCopy();
        }
        else
        {
            for (unsigned i=0; i<n; ++i)
                *this += (f*r[i]).theCopy();
        }
    }

    void LikelihoodAccumulator::accSimpleCurve(
        const AbsLogLikelihoodCurve& r)
    {
        components_.emplace_back(r);
        tagModified();
    }

    double LikelihoodAccumulator::operator()(const double x) const
    {
        long double sum = 0.0L;
        const unsigned n = components_.size();
        for (unsigned i=0; i<n; ++i)
            sum += components_[i](x);
        return factor_*sum;
    }

    double LikelihoodAccumulator::derivative(const double x) const
    {
        long double sum = 0.0L;
        const unsigned n = components_.size();
        for (unsigned i=0; i<n; ++i)
            sum += components_[i].derivative(x);
        return factor_*sum;
    }

    double LikelihoodAccumulator::secondDerivative(
        const double x, const double step) const
    {
        long double sum = 0.0L;
        const unsigned n = components_.size();
        for (unsigned i=0; i<n; ++i)
            sum += components_[i].secondDerivative(x, step);
        return factor_*sum;
    }

    void LikelihoodAccumulator::calcSupport() const
    {
        Interval<double> overlap(-DBL_MAX, DBL_MAX, CLOSED_INTERVAL);
        const unsigned n = components_.size();
        for (unsigned i=0; i<n; ++i)
        {
            const Interval<double> support(components_[i].parMin(),
                                           components_[i].parMax(),
                                           CLOSED_INTERVAL);
            overlap = overlap.overlap(support);
        }
        if (overlap.length() == 0.0) throw std::runtime_error(
            "In ase::LikelihoodAccumulator::calcSupport: "
            "empty support interval");
        pmin_ = overlap.min();
        pmax_  = overlap.max();
        supportCalculated_ = true;
    }

    void LikelihoodAccumulator::calcLocation() const
    {
        const unsigned n = components_.size();
        if (n)
        {
            if (!maximumFound_)
            {
                try {findMaximum();}
                catch (const std::runtime_error&) {}
            }
            if (maximumFound_)
                loc_ = maxArg_;
            else
                loc_ = averageLocation();
        }
        else
            loc_ = 0.0;
        locationCalculated_ = true;
    }

    double LikelihoodAccumulator::averageLocation() const
    {
        const unsigned n = components_.size();
        assert(n);
        long double sum = 0.0L;
        for (unsigned i=0; i<n; ++i)
            sum += components_[i].location();
        return sum/n;
    }

    void LikelihoodAccumulator::calcStepSize() const
    {
        const unsigned n = components_.size();
        if (n)
        {
            stepSz_ = components_[0].stepSize();
            for (unsigned i=1U; i<n; ++i)
            {
                const double step = components_[i].stepSize();
                if (step < stepSz_)
                    stepSz_ = step;
            }
        }
        else
            stepSz_ = 1.0;
        stepCalculated_ = true;
    }

    void LikelihoodAccumulator::findMaximum() const
    {
        static const unsigned maxSteps = 2000;
        static const double stepFactor = 1.1;

        const unsigned n = components_.size();
        if (n)
        {
            if (n == 1U && factor_ > 0.0)
            {
                maxArg_ = components_[0].argmax();
                maxValue_ = (*this)(maxArg_);
            }
            else
            {
                // Try to find the maximum starting from a variety of points
                std::vector<double> starts;
                starts.reserve(n + 2U);
                for (unsigned i=0; i<n; ++i)
                    starts.push_back(components_[i].argmax());
                starts.push_back(averageLocation());
                starts.push_back((parMin() + parMax())/2.0);
                std::sort(starts.begin(), starts.end());
                auto last = std::unique(starts.begin(), starts.end());
                starts.erase(last, starts.end());

                const unsigned nStarts = starts.size();
                std::vector<std::pair<double,double> > results(nStarts);
                for (unsigned i=0; i<nStarts; ++i)
                    results[i] = findLocalMaximum(
                        starts[i], derivative(starts[i]) >= 0.0,
                        maxSteps, stepFactor);
                if (nStarts > 1U)
                {
                    GreaterBySecondDD g2;
                    std::sort(results.begin(), results.end(), g2);
                }
                if (results[0].second == -DBL_MAX) throw std::runtime_error(
                    "In ase::LikelihoodAccumulator::findMaximum: failed to "
                    "find the maximum. Please ensure that the curve is concave.");
                maxArg_ = results[0].first;
                maxValue_ = results[0].second;
            }
        }
        else
        {
            maxValue_ = 0.0;
            maxArg_ = 0.0;
        }
        maximumFound_ = true;
    }
}
