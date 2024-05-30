#include <sstream>

#include "ase/statUtils.hh"

namespace ase {
    void validateSkewnessKurtosis(const double skewness,
                                  const double kurtosis)
    {
        const double b1 = skewness*skewness;
        if (kurtosis < b1 + 1.0)
        {
            std::ostringstream os;
            os.precision(17);
            os << "In ase::validateSkewnessKurtosis: "
               << "combination of skewness = " << skewness
               << " and kurtosis = " << kurtosis
               << " is impossible";
            throw std::invalid_argument(os.str());
        }
    }

    void momentsToCumulants(const long double* const moms,
                            long double* const out,
                            const unsigned order)
    {
        if (order > 4U) throw std::invalid_argument(
            "In ase::momentsToCumulants: "
            "conversion is implemented to 4th order only");
        if (order)
        {
            assert(moms);
            assert(out);
            long double cums[4];

            // Mean
            const long double mu = moms[0];
            const long double muSq = mu*mu;
            cums[0] = mu;
            if (order > 1U)
                // Variance
                cums[1] = moms[1] - muSq;
            if (order > 2U)
                // Skewness
                cums[2] = moms[2] + mu*(2.0*muSq - 3.0*moms[1]);
            if (order > 3U)
                // Kurtosis
                cums[3] = moms[3] - 4.0*mu*moms[2] + 3.0*muSq*(2.0*moms[1] - muSq) - 3.0*cums[1]*cums[1];

            for (unsigned i=0; i<order; ++i)
                out[i] = cums[i];
        }
    }

    void cumulantsToMoments(const long double* const cums,
                            long double* const out,
                            const unsigned order)
    {
        if (order > 4U) throw std::invalid_argument(
            "In ase::cumulantsToMoments: "
            "conversion is implemented to 4th order only");
        if (order)
        {
            assert(cums);
            assert(out);
            long double moms[4];

            const long double mu = cums[0];
            const long double muSq = mu*mu;
            moms[0] = mu;
            if (order > 1U)
                moms[1] = cums[1] + muSq;
            if (order > 2U)
                moms[2] = cums[2] - mu*(2.0*muSq - 3.0*moms[1]);
            if (order > 3U)
                moms[3] = cums[3] + 4.0*mu*moms[2] - 3.0*muSq*(2.0*moms[1] - muSq) + 3.0*cums[1]*cums[1];

            for (unsigned i=0; i<order; ++i)
                out[i] = moms[i];
        }
    }
}
