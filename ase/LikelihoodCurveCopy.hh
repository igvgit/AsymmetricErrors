#ifndef ASE_LIKELIHOODCURVECOPY_HH_
#define ASE_LIKELIHOODCURVECOPY_HH_

#include "ase/AbsLogLikelihoodCurve.hh"

namespace ase {
    /** Class to manage AbsLogLikelihoodCurve copies */
    class LikelihoodCurveCopy : public AbsLogLikelihoodCurve
    {
    public:
        /** This constructor assumes the ownership of the pointer */
        inline LikelihoodCurveCopy(AbsLogLikelihoodCurve* ptr)
            : copy_(ptr) {assert(ptr);}

        inline LikelihoodCurveCopy(const AbsLogLikelihoodCurve& r)
            : copy_(0)
        {
            const LikelihoodCurveCopy* c =
                dynamic_cast<const LikelihoodCurveCopy*>(&r);
            if (c)
                copy_ = c->copy_->clone();
            else
                copy_ = r.clone();
        }

        inline LikelihoodCurveCopy(const LikelihoodCurveCopy& r)
            : copy_(r.copy_->clone())
        {
        }

        inline LikelihoodCurveCopy(LikelihoodCurveCopy&& r)
            : copy_(r.copy_)
        {
            r.copy_ = 0;
        }

        inline LikelihoodCurveCopy& operator=(const LikelihoodCurveCopy& r)
        {
            if (this != &r)
            {
                AbsLogLikelihoodCurve* tmp = r.copy_->clone();
                delete copy_;
                copy_ = tmp;
            }
            return *this;
        }

        inline LikelihoodCurveCopy& operator=(LikelihoodCurveCopy&& r)
        {
            if (this != &r)
            {
                delete copy_;
                copy_ = r.copy_;
                r.copy_ = 0;
            }
            return *this;
        }

        inline LikelihoodCurveCopy& operator=(const AbsLogLikelihoodCurve& r)
        {
            AbsLogLikelihoodCurve* tmp = 0;
            const LikelihoodCurveCopy* c =
                dynamic_cast<const LikelihoodCurveCopy*>(&r);
            if (c)
            {
                if (this != c)
                    tmp = c->copy_->clone();
            }
            else
                tmp = r.clone();
            if (tmp)
            {
                delete copy_;
                copy_ = tmp;
            }
            return *this;
        }

        inline virtual LikelihoodCurveCopy* clone() const override
            {return new LikelihoodCurveCopy(*this);}

        inline virtual ~LikelihoodCurveCopy() override {delete copy_;}

        inline const AbsLogLikelihoodCurve& theCopy() const
            {return *copy_;}

        inline virtual double parMin() const override
            {return copy_->parMin();}

        inline virtual double parMax() const override
            {return copy_->parMax();}

        inline virtual double location() const override
            {return copy_->location();}

        inline virtual double stepSize() const override
            {return copy_->stepSize();}

        inline virtual double maximum() const override
            {return copy_->maximum();}

        inline virtual double argmax() const override
            {return copy_->argmax();}

        inline virtual double operator()(const double p) const override
            {return (*copy_)(p);}

        inline virtual double derivative(const double p) const override
            {return copy_->derivative(p);}

        inline virtual double secondDerivative(
            const double p, const double step = 0.0) const override
            {return copy_->secondDerivative(p, step);}

        inline virtual std::string classname() const override
            {return copy_->classname();}

        inline virtual AbsLogLikelihoodCurve& operator*=(const double c) override
            {*copy_ *= c; return *this;}

        inline virtual AbsLogLikelihoodCurve& operator/=(const double c) override
            {*copy_ /= c; return *this;}

        inline virtual double sigmaPlus(const double deltaLogLikelihood = 0.5,
                                        const double stepFactor = 1.1) const override
            {return copy_->sigmaPlus(deltaLogLikelihood, stepFactor);}

        inline virtual double sigmaMinus(const double deltaLogLikelihood = 0.5,
                                         const double stepFactor = 1.1) const override
            {return copy_->sigmaMinus(deltaLogLikelihood, stepFactor);}

        inline virtual std::pair<double,double> findLocalMaximum(
            const double startingPoint, const bool searchToTheRight,
            const unsigned maxSteps, const double stepFactor = 1.1) const override
            {return copy_->findLocalMaximum(startingPoint, searchToTheRight,
                                            maxSteps, stepFactor);}

        inline virtual double posteriorMean() const override
            {return copy_->posteriorMean();}
        
        inline virtual double posteriorVariance() const override
            {return copy_->posteriorVariance();}

    protected:
        AbsLogLikelihoodCurve* copy_;
    };
}

// Binary operators
inline ase::LikelihoodCurveCopy operator*(
    const ase::AbsLogLikelihoodCurve& l, const double& c)
{
    ase::LikelihoodCurveCopy copy(l);
    copy *= c;
    return copy;
}

inline ase::LikelihoodCurveCopy operator*(
    const double& c, const ase::AbsLogLikelihoodCurve& r)
{
    ase::LikelihoodCurveCopy copy(r);
    copy *= c;
    return copy;
}

inline ase::LikelihoodCurveCopy operator/(
    const ase::AbsLogLikelihoodCurve& l, const double& c)
{
    ase::LikelihoodCurveCopy copy(l);
    copy /= c;
    return copy;
}

ase::LikelihoodCurveCopy operator+(
    const ase::AbsLogLikelihoodCurve& l, const ase::AbsLogLikelihoodCurve& r);

ase::LikelihoodCurveCopy operator-(
    const ase::AbsLogLikelihoodCurve& l, const ase::AbsLogLikelihoodCurve& r);

#endif // ASE_LIKELIHOODCURVECOPY_HH_
