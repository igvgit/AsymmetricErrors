#ifndef ASE_ARRAYSTATS_HH_
#define ASE_ARRAYSTATS_HH_

#include <cassert>
#include <stdexcept>
#include <vector>

namespace ase {
    /**
    // Function for calculating multiple sample cumulants (k-statistics).
    // The array "cumulants" should have at least "maxOrder+1" elements.
    // Upon exit, cumulants[k] will be set to the sample cumulant of order k.
    // Currently, "maxOrder" argument can not exceed 6. Formulae for these
    // can be found, for example, in "Kendall's Advanced Theory of Statistics"
    // by Stuart and Ord.
    */
    template<typename Numeric>
    void arrayCumulants(const Numeric* arr, const unsigned long sz,
                        const unsigned maxOrder, long double* cumulants)
    {
        const unsigned orderSupported = 6U;
        long double s[orderSupported+1U] = {0.0L};

        if (maxOrder > orderSupported)
            throw std::invalid_argument("In ase::arrayCumulants: order argument "
                                        "outside of supported range");
        if (!sz || sz < maxOrder)
            throw std::invalid_argument("In ase::arrayCumulants: "
                                        "insufficient array length");
        assert(arr);
        assert(cumulants);
        cumulants[0] = 1.0L;

        if (maxOrder)
        {
            long double mean = 0.0L;
            for (unsigned long i=0; i<sz; ++i)
                mean += static_cast<long double>(arr[i]);
            mean /= sz;
            cumulants[1] = mean;

            if (maxOrder > 1U)
            {
                for (unsigned long i=0; i<sz; ++i)
                {
                    const long double delta = static_cast<long double>(arr[i]) - mean;
                    long double dk = delta;
                    for (unsigned k=2U; k<=maxOrder; ++k)
                    {
                        dk *= delta;
                        s[k] += dk;
                    }
                }
                cumulants[2] = s[2]/(sz - 1U);

                if (maxOrder > 2U)
                    cumulants[3] = sz*s[3]/(sz-1U)/(sz-2U);

                if (maxOrder > 3U)
                {
                    const long double n = sz;
                    const long double n2 = n*n;
                    cumulants[4] = ((n2 + n)*s[4] - s[2]*s[2]*3*(sz - 1U))/(sz-1U)/(sz-2U)/(sz-3U);

                    if (maxOrder > 4U)
                    {
                        const long double n3 = n2*n;
                        cumulants[5] = ((n3 + 5*n2)*s[5] - s[3]*s[2]*10*(n2 - n))/(sz-1U)/(sz-2U)/(sz-3U)/(sz-4U);

                        if (maxOrder > 5U)
                        {
                            const long double n4 = n2*n2;
                            cumulants[6] = ((n4 + 16*n3 + 11*n2 - 4*n)*s[6]
                                            - 15*(n-1)*(n-1)*(n+4)*s[4]*s[2]
                                            - 10*(n3 - 2*n2 + 5*n - 4)*s[3]*s[3]
                                            + 30*(n2 - 3*n + 2)*s[2]*s[2]*s[2])/(sz-1U)/(sz-2U)/(sz-3U)/(sz-4U)/(sz-5U);
                        }
                    }
                }
            }
        }
    }

#ifdef SWIG
    // The Python version will skip the 0th order cumulant.
    // It will also be renamed into just "arrayCumulants".
    inline std::vector<double>
    swigArrayCumulants(const double* arr, const unsigned sz,
                       const unsigned maxOrder)
    {
        std::vector<long double> tmp(maxOrder + 1U);
        arrayCumulants(arr, sz, maxOrder, &tmp[0]);
        std::vector<double> result(tmp.begin()+1U, tmp.end());
        return result;
    }
#endif // SWIG
}

#endif // ASE_ARRAYSTATS_HH_
