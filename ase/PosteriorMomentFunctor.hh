#ifndef ASE_POSTERIORMOMENTFUNCTOR_HH_
#define ASE_POSTERIORMOMENTFUNCTOR_HH_

#include <cmath>

#include "ase/AbsLogLikelihoodCurve.hh"

namespace ase {
    class PosteriorMomentFunctor
    {
    public:
        inline PosteriorMomentFunctor(const AbsLogLikelihoodCurve& fcn,
                                      const double p0, const unsigned n)
            : fcn_(fcn), p0_(p0), n_(n) {}

        inline double operator()(const double x) const
        {
            double deltaPow = 1.0;
            if (n_)
            {
                const double delta = x - p0_;
                switch (n_)
                {
                case 1U:
                    deltaPow = delta;
                    break;
                case 2U:
                    deltaPow = delta*delta;
                    break;
                case 3U:
                    deltaPow = delta*delta*delta;
                    break;
                case 4U:
                    {
                        const double delta2 = delta*delta;
                        deltaPow = delta2*delta2;
                    }
                    break;
                default:
                    deltaPow = std::pow(delta, n_);
                }
            }
            return deltaPow*std::exp(fcn_(x));
        }

    private:
        const AbsLogLikelihoodCurve& fcn_;
        double p0_;
        unsigned n_;
    };
}

#endif // ASE_POSTERIORMOMENTFUNCTOR_HH_
