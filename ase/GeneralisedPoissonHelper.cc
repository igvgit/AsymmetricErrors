#include <cmath>
#include <algorithm>
#include <utility>
#include <cfloat>
#include <cassert>

#include "ase/GeneralisedPoissonHelper.hh"
#include "ase/findRootUsingBisections.hh"
#include "ase/PosteriorMomentFunctor.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace {
    class EqFCN
    {
    public:
        inline EqFCN(const double r)
            : r_(r) {assert(r_ > 1.0);}

        inline double operator()(const double t) const
            {return (1.0 - t)/(1 + r_*t) - exp(-t*(1.0 + r_));}

    private:
        double r_;
    };
}

// Find the positive solution of the equation
// (1 - t)/(1 + r*t) == exp(-t*(1 + r)).
// Such a solution exists if r > 1.
static double getTheT(const double r)
{
    const EqFCN eqFcn(r);
    double tryT = 1.0;
    double fcn = eqFcn(tryT);
    while (fcn < 0.0)
    {
        tryT /= 2.0;
        fcn = eqFcn(tryT);
    }
    assert(tryT > 0.0);
    if (fcn == 0.0)
        return tryT;
    else
    {
        double t;
        const bool status = ase::findRootUsingBisections(
            eqFcn, 0.0, tryT, tryT*2.0, 2.0*DBL_EPSILON, &t);
        assert(status);
        return t;
    }
}

namespace ase {
    GeneralisedPoissonHelper::GeneralisedPoissonHelper(
        const double mu, const double sigPlus, const double sigMinus)
        : AbsShiftableLogli(mu),
          sigmaBig_(sigPlus), sigmaSmall_(sigMinus),
          pmin_(-DBL_MAX), pmax_(DBL_MAX), alpha_(0.0), nu_(0.0),
          isSymmetric_(sigmaBig_ == sigmaSmall_)
    {
        validateSigmas("ase::GeneralisedPoissonHelper constructor",
                       sigPlus, sigMinus);
        if (sigmaBig_ < sigmaSmall_)
            std::swap(sigmaBig_, sigmaSmall_);
        if (!isSymmetric_)
        {
            const double t = getTheT(sigmaBig_/sigmaSmall_);
            const double gamma = t/sigmaSmall_;
            const double gammasb = gamma*sigmaBig_;
            nu_ = 0.5/(gammasb - log(1.0 + gammasb));
            alpha_ = gamma*nu_;
            pmax_ = DBL_MAX;
            pmin_ = -(1.0 - DBL_EPSILON)/gamma;
        }
    }

    double GeneralisedPoissonHelper::stepSize() const
    {
        return 0.01*sigmaSmall_;
    }

    double GeneralisedPoissonHelper::uValue(const double x) const
    {
        if (isSymmetric_)
        {
            const double del = x/sigmaBig_;
            return -del*del/2.0;
        }
        else
            return nu_*log(1.0 + alpha_/nu_*x) - alpha_*x;
    }

    double GeneralisedPoissonHelper::uDerivative(const double x) const
    {
        if (isSymmetric_)
            return -x/sigmaBig_/sigmaBig_;
        else
        {
            const double anux = alpha_/nu_*x;
            return -alpha_*anux/(1.0 + anux);
        }
    }

    double GeneralisedPoissonHelper::uSecondDerivative(const double x, double) const
    {
        if (isSymmetric_)
            return -1.0/sigmaBig_/sigmaBig_;
        else
        {
            const double tmp = nu_ + alpha_*x;
            return -alpha_*alpha_*nu_/tmp/tmp;
        }
    }

    double GeneralisedPoissonHelper::uSigmaPlus(const double deltaLogLikelihood,
                                                const double stepFactor) const
    {
        if (isSymmetric_)
            return sqrt(2.0*deltaLogLikelihood)*sigmaBig_;
        else
            return AbsLogLikelihoodCurve::sigmaPlus(
                deltaLogLikelihood*factor(), stepFactor);
    }

    double GeneralisedPoissonHelper::uSigmaMinus(const double deltaLogLikelihood,
                                                 const double stepFactor) const
    {
        if (isSymmetric_)
            return sqrt(2.0*deltaLogLikelihood)*sigmaBig_;
        else
            return AbsLogLikelihoodCurve::sigmaMinus(
                deltaLogLikelihood*factor(), stepFactor);
    }
}
