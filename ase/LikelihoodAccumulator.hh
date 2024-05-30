#ifndef ASE_LIKELIHOODACCUMULATOR_HH_
#define ASE_LIKELIHOODACCUMULATOR_HH_

#include <vector>

#include "ase/AbsLogLikelihoodCurve.hh"
#include "ase/LikelihoodCurveCombination.hh"

namespace ase {
    /** A class for combining multiple likelihood curves */
    class LikelihoodAccumulator : public AbsLogLikelihoodCurve
    {
    public:
        LikelihoodAccumulator();

        inline virtual ~LikelihoodAccumulator() override {}

        inline virtual LikelihoodAccumulator* clone() const override
            {return new LikelihoodAccumulator(*this);}

        inline bool empty() const {return components_.empty();}
        inline unsigned size() const {return components_.size();}

        // Don't allow outside code to modify stored curves
        inline const LikelihoodCurveCopy& operator[](const unsigned i) const
            {return components_[i];}
        inline const LikelihoodCurveCopy& at(const unsigned i) const
            {return components_.at(i);}

        // Get a modifiable copy
        inline LikelihoodCurveCopy getCurve(const unsigned i) const
            {return components_.at(i);}

        inline double factor() const {return factor_;}

        // Main accumulator method
        virtual void accumulate(const AbsLogLikelihoodCurve& r);

        // Accumulator operators
        inline LikelihoodAccumulator& operator+=(const AbsLogLikelihoodCurve& r)
            {this->accumulate(r); return *this;}
        inline LikelihoodAccumulator& operator-=(const AbsLogLikelihoodCurve& r)
            {this->accumulate((-1.0)*r); return *this;}

        // Methods to override from AbsLogLikelihoodCurve
        inline virtual double parMin() const override
        {
            if (!supportCalculated_)
                calcSupport();
            return pmin_;
        }

        inline virtual double parMax() const override
        {
            if (!supportCalculated_)
                calcSupport();
            return pmax_;
        }

        inline virtual double location() const override
        {
            if (!locationCalculated_)
                calcLocation();
            return loc_;
        }

        inline virtual double stepSize() const override
        {
            if (!stepCalculated_)
                calcStepSize();
            return stepSz_;
        }

        inline virtual double maximum() const override
        {
            if (!maximumFound_)
                findMaximum();
            return maxValue_;
        }

        inline virtual double argmax() const override
        {
            if (!maximumFound_)
                findMaximum();
            return maxArg_;
        }

        virtual double operator()(double parameter) const override;
        virtual double derivative(double parameter) const override;
        virtual double secondDerivative(
            double parameter, double step = 0.0) const override;

        inline virtual std::string classname() const override
            {return "LikelihoodAccumulator";}

        inline virtual AbsLogLikelihoodCurve& operator*=(const double c) override
            {if (c != 1.0) {factor_ *= c; tagModified();} return *this;}

    private:
        struct GreaterBySecondDD
        {
            inline bool operator()(const std::pair<double,double>& x,
                                   const std::pair<double,double>& y) const
                {return y.second < x.second;}
        };

        void tagModified();

        void accLikelihoodCurveSum(const LikelihoodCurveSum& r);
        void accLikelihoodCurveDifference(const LikelihoodCurveDifference& r);
        void accLikelihoodAccumulator(const LikelihoodAccumulator& r);
        void accSimpleCurve(const AbsLogLikelihoodCurve& r);

        void calcSupport() const;
        void calcLocation() const;
        void calcStepSize() const;
        void findMaximum() const;

        double averageLocation() const;

        std::vector<LikelihoodCurveCopy> components_;
        double factor_;
        mutable double pmin_;
        mutable double pmax_;
        mutable double loc_;
        mutable double stepSz_;
        mutable double maxValue_;
        mutable double maxArg_;
        mutable bool supportCalculated_;
        mutable bool locationCalculated_;
        mutable bool stepCalculated_;
        mutable bool maximumFound_;
    };
}

#endif // ASE_LIKELIHOODACCUMULATOR_HH_
