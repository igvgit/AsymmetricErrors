#include <cmath>
#include <cassert>

#include "ase/PoissonLogli.hh"

namespace ase {
    double PoissonLogli::operator()(const double lam) const
    {
        if (n_)
        {
            assert(lam > 0.0);
            const double r = lam/dn_;
            return factor_*dn_*(log(r) + 1.0 - r);
        }
        else
        {
            assert(lam >= 0.0);
            return -lam*factor_;
        }
    }

    double PoissonLogli::derivative(const double lam) const
    {
        if (n_)
        {
            assert(lam > 0.0);
            return factor_*(dn_/lam - 1.0);
        }
        else
        {
            assert(lam >= 0.0);
            return -factor_;
        }
    }

    double PoissonLogli::secondDerivative(
        const double lam, double /* step */) const
    {
        if (n_)
        {
            assert(lam > 0.0);
            return -factor_*dn_/lam/lam;
        }
        else
        {
            assert(lam >= 0.0);
            return 0.0;
        }
    }
}
