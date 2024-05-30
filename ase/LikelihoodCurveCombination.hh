#ifndef ASE_LIKELIHOODCURVECOMBINATION_HH_
#define ASE_LIKELIHOODCURVECOMBINATION_HH_

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cfloat>
#include <utility>

#include "ase/LikelihoodCurveCopy.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/Interval.hh"

namespace ase {
    namespace Private {
        template <class T> struct Adding {enum {value = 0};};
        template <> struct Adding<std::plus<double> > {enum {value = 1};};
    }

    /** A class for adding or subtracting two likelihood curves */
    template<class BinaryOp>
    class LikelihoodCurveCombination : public AbsLogLikelihoodCurve
    {
    public:
        inline LikelihoodCurveCombination(const AbsLogLikelihoodCurve& l,
                                          const AbsLogLikelihoodCurve& r)
            : left_(l), right_(r), pmin_(0.0), pmax_(0.0), factor_(1.0),
              logliMax_(0.0), argmax_(0.0), location_(0.0),
              maxFound_(false), locFound_(false)
        {
            const Interval<double> support1(l.parMin(), l.parMax(), OPEN_INTERVAL);
            const Interval<double> support2(r.parMin(), r.parMax(), OPEN_INTERVAL);
            const Interval<double>& over = support1.overlap(support2);
            if (over.empty()) throw std::invalid_argument(
                "In ase::LikelihoodCurveCombination constructor: "
                "likelihood curve supports do not overlap");
            pmin_ = over.min();
            pmax_ = over.max();
        }

        inline virtual ~LikelihoodCurveCombination() override {}

        inline virtual LikelihoodCurveCombination* clone() const override
            {return new LikelihoodCurveCombination(*this);}

        inline bool adding() const
            {return Private::Adding<BinaryOp>::value;}

        inline const LikelihoodCurveCopy& leftOperand() const
            {return left_;}

        inline const LikelihoodCurveCopy& rightOperand() const
            {return right_;}

        inline double factor() const {return factor_;}

        inline virtual double parMin() const override
            {return pmin_;}

        inline virtual double parMax() const override
            {return pmax_;}

        inline virtual double location() const override
        {
            if (!locFound_)
                findLocation();
            return location_;
        }

        inline virtual double stepSize() const override
            {return std::min(left_.stepSize(), right_.stepSize());}

        inline virtual double maximum() const override
        {
            if (!maxFound_)
                findMaximum();
            return logliMax_;
        }

        inline virtual double argmax() const override
        {
            if (!maxFound_)
                findMaximum();
            return argmax_;
        }

        inline virtual double operator()(const double p) const override
            {return factor_*op_(left_(p), right_(p));}

        inline virtual double derivative(const double p) const override
            {return factor_*op_(left_.derivative(p), right_.derivative(p));}

        inline virtual double secondDerivative(
            const double p, const double step = 0.0) const override
            {return factor_*op_(left_.secondDerivative(p, step),
                                right_.secondDerivative(p, step));}

        inline virtual std::string classname() const override
            {return adding() ? "LikelihoodCurveSum" : "LikelihoodCurveDifference";}

        inline virtual AbsLogLikelihoodCurve& operator*=(const double c) override
        {
            if (c != 1.0)
            {
                factor_ *= c;
                if (c > 0.0 && factor_ && maxFound_)
                    logliMax_ *= c;
                else
                {
                    logliMax_ = 0.0;
                    argmax_  = 0.0;
                    maxFound_ = false;
                }
            }
            return *this;
        }

    private:
        struct GreaterBySecondDD
        {
            inline bool operator()(const std::pair<double,double>& x,
                                   const std::pair<double,double>& y) const
                {return y.second < x.second;}
        };

        inline void findMaximum() const
        {
            static const unsigned maxSteps = 2000;

            const double argmaxL = left_.location();
            const double argmaxR = right_.location();
            if (argmaxL == argmaxR && adding() && factor_ > 0.0)
                if (argmaxL == left_.argmax() && argmaxL == right_.argmax())
                {
                    argmax_ = argmaxL;
                    logliMax_ = (*this)(argmax_);
                    maxFound_ = true;
                    return;
                }

            // Try several different starting points to search for maximum
            double starts[4];
            unsigned nStarts = 0;
            starts[nStarts++] = argmaxL;
            const double pmid = (pmin_ + pmax_)/2.0;
            if (argmaxL != pmid)
                starts[nStarts++] = pmid;
            if (argmaxL != argmaxR)
            {
                starts[nStarts++] = argmaxR;
                starts[nStarts++] = (argmaxL + argmaxR)/2.0;
            }
            assert(nStarts <= sizeof(starts)/sizeof(starts[0]));

            std::pair<double,double> results[sizeof(starts)/sizeof(starts[0])];
            for (unsigned i=0; i<nStarts; ++i)
                results[i] = findLocalMaximum(
                    starts[i], derivative(starts[i]) >= 0.0, maxSteps);
            if (nStarts > 1U)
            {
                GreaterBySecondDD g2;
                std::sort(results, results+nStarts, g2);
            }
            if (results[0].second == -DBL_MAX) throw std::runtime_error(
                "In ase::LikelihoodCurveCombination::findMaximum: failed to "
                "find the maximum. Please ensure that the curve is concave.");
            argmax_ = results[0].first;
            logliMax_ = results[0].second;
            maxFound_ = true;
        }

        inline void findLocation() const
        {
            if (!maxFound_)
            {
                try {findMaximum();}
                catch (const std::runtime_error&) {}
            }
            if (maxFound_)
                location_ = argmax_;
            else
                location_ = (left_.location() + right_.location())/2.0;
            locFound_ = true;
        }

        LikelihoodCurveCopy left_;
        LikelihoodCurveCopy right_;
        double pmin_;
        double pmax_;
        double factor_;
        BinaryOp op_;
        mutable double logliMax_;
        mutable double argmax_;
        mutable double location_;
        mutable bool maxFound_;
        mutable bool locFound_;
    };

    typedef LikelihoodCurveCombination<std::plus<double> > LikelihoodCurveSum;
    typedef LikelihoodCurveCombination<std::minus<double> > LikelihoodCurveDifference;
}

#endif // ASE_LIKELIHOODCURVECOMBINATION_HH_
