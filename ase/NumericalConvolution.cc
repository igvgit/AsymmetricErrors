#include <cmath>
#include <stdexcept>
#include <algorithm>

#include "ase/NumericalConvolution.hh"
#include "ase/DistributionFunctors1D.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace ase {
    NumericalConvolution::NumericalConvolution(
        const AbsDistributionModel1D& m1, const AbsDistributionModel1D& m2,
        const unsigned nPt, const unsigned nIntervals)
        : f1_(m1), quantiles_(nPt*nIntervals),
          weights_(nPt*nIntervals), buf_(nPt*nIntervals)
    {
        if (!nIntervals) throw std::invalid_argument(
            "In ase::NumericalConvolution constructor: "
            "number of integration intervals must be positive");
        const GaussLegendreQuadrature glq(nPt);
        long double* absc = &weights_[0];
        glq.getAllAbscissae(absc, nPt);
        if (nIntervals == 1U)
        {
            for (unsigned i=0; i<nPt; ++i)
            {
                const double u = (absc[i] + 1.0)/2.0;
                quantiles_[i] = m2.quantile(u);
            }
            unit_ = 0.5;
        }
        else
        {
            double xmin = 0.0;
            double* q = &quantiles_[0];
            for (unsigned interN=0; interN<nIntervals; ++interN)
            {
                const double xmax = static_cast<double>(interN + 1U)/nIntervals;
                const double midpoint = (xmin + xmax)/2.0L;
                const double unit = (xmax - xmin)/2.0L;
                for (unsigned i=0; i<nPt; ++i)
                {
                    const double u = midpoint + unit*absc[i];
                    *q++ = m2.quantile(u);
                }
                xmin = xmax;
            }
            unit_ = 0.5/nIntervals;
        }
        for (unsigned interN=0; interN<nIntervals; ++interN)
            glq.getAllWeights(absc + interN*nPt, nPt);
    }

    double NumericalConvolution::operator()(const double x) const
    {
        const unsigned sz = quantiles_.size();
        for (unsigned i=0; i<sz; ++i)
        {
            const double d = weights_[i]*f1_.density(x - quantiles_[i]);
            buf_[i] = d;
        }
        std::sort(buf_.begin(), buf_.end());
        long double acc = 0.0L;
        for (unsigned i=0; i<sz; ++i)
            acc += buf_[i];
        return acc*unit_;
    }
}
