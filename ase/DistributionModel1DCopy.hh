#ifndef ASE_DISTRIBUTIONMODEL1DCOPY_HH_
#define ASE_DISTRIBUTIONMODEL1DCOPY_HH_

#include <cassert>

#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    /**
    // Sometimes we need a copy of AbsDistributionModel1D object
    // instead of just a reference to it. Such a copy can always
    // be created with the clone() method, but this method returns
    // a pointer to the heap object that has to be managed. This
    // class takes care of the necessary memory management, while
    // simultaneously acting as AbsDistributionModel1D.
    */
    class DistributionModel1DCopy : public AbsDistributionModel1D
    {
    public:
        /** This constructor assumes the ownership of the pointer */
        inline DistributionModel1DCopy(const AbsDistributionModel1D* ptr)
            : copy_(ptr) {assert(ptr);}

        inline DistributionModel1DCopy(const AbsDistributionModel1D& r)
            : copy_(0)
        {
            const DistributionModel1DCopy* c =
                dynamic_cast<const DistributionModel1DCopy*>(&r);
            if (c)
                copy_ = c->copy_->clone();
            else
                copy_ = r.clone();
        }

        inline DistributionModel1DCopy(const DistributionModel1DCopy& r)
            : copy_(r.copy_->clone())
        {
        }

        inline DistributionModel1DCopy(DistributionModel1DCopy&& r)
            : copy_(r.copy_)
        {
            r.copy_ = 0;
        }

        inline DistributionModel1DCopy& operator=(const DistributionModel1DCopy& r)
        {
            if (this != &r)
            {
                const AbsDistributionModel1D* tmp = r.copy_->clone();
                delete copy_;
                copy_ = tmp;
            }
            return *this;
        }

        inline DistributionModel1DCopy& operator=(DistributionModel1DCopy&& r)
        {
            if (this != &r)
            {
                delete copy_;
                copy_ = r.copy_;
                r.copy_ = 0;
            }
            return *this;
        }

        inline DistributionModel1DCopy& operator=(const AbsDistributionModel1D& r)
        {
            const AbsDistributionModel1D* tmp = 0;
            const DistributionModel1DCopy* c =
                dynamic_cast<const DistributionModel1DCopy*>(&r);
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

        inline virtual DistributionModel1DCopy* clone() const override
            {return new DistributionModel1DCopy(*this);}

        inline virtual ~DistributionModel1DCopy() override {delete copy_;}

        inline const AbsDistributionModel1D& theCopy() const
            {return *copy_;}

        inline virtual double density(const double x) const override
            {return copy_->density(x);}

        inline virtual bool isDensityContinuous() const override
            {return copy_->isDensityContinuous();}

        inline virtual bool isNonNegative() const override
            {return copy_->isNonNegative();}

        inline virtual bool isUnimodal() const override
            {return copy_->isUnimodal();}

        inline virtual double densityDerivative(const double x) const override
            {return copy_->densityDerivative(x);}

        inline virtual double cdf(const double x) const override
            {return copy_->cdf(x);}

        inline virtual double exceedance(const double x) const override
            {return copy_->exceedance(x);}

        inline virtual double quantile(const double x) const override
            {return copy_->quantile(x);}

        inline virtual double invExceedance(const double x) const override
            {return copy_->invExceedance(x);}

        inline virtual double cumulant(const unsigned n) const override
            {return copy_->cumulant(n);}

        inline virtual double mode() const override
            {return copy_->mode();}

        inline virtual double descentDelta(
            const bool isToTheRight, const double deltaLnL=0.5) const override
            {return copy_->descentDelta(isToTheRight, deltaLnL);}

        inline virtual std::string classname() const override
            {return copy_->classname();}

        inline virtual double random(AbsRNG& gen) const override
            {return copy_->random(gen);}

        inline virtual double qWidth() const override
            {return copy_->qWidth();}

        inline virtual double qAsymmetry() const override
            {return copy_->qAsymmetry();}

    protected:
        const AbsDistributionModel1D* copy_;
    };
}

#endif // ASE_DISTRIBUTIONMODEL1DCOPY_HH_
