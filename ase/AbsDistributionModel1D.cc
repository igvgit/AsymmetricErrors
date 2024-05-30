#include "ase/gCdfValues.hh"
#include "ase/AbsDistributionModel1D.hh"

namespace ase {
    double AbsDistributionModel1D::random(AbsRNG& gen) const
    {
        // The following should, in principle, be able to sample
        // the tails better than simply calling quantile(gen())
        if (gen() > 0.5)
            return invExceedance(gen());
        else
            return quantile(gen());
    }

    double AbsDistributionModel1D::qWidth() const
    {
        const double q16 = this->quantile(GCDF16);
        const double q84 = this->quantile(GCDF84);
        return (q84 - q16)/2.0;
    }

    double AbsDistributionModel1D::qAsymmetry() const
    {
        const double q16 = this->quantile(GCDF16);
        const double med = this->quantile(0.5);
        const double q84 = this->quantile(GCDF84);
        const double sp = q84 - med;
        const double sm = med - q16;
        return (sp - sm)/(sp + sm);
    }
}
