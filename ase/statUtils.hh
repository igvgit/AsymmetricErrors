#ifndef ASE_STATUTILS_HH_
#define ASE_STATUTILS_HH_

#include <cassert>
#include <stdexcept>

namespace ase {
    // Throw std::invalid_argument in case the given combination
    // of skewness and kurtosis is impossible
    void validateSkewnessKurtosis(double skewness, double kurtosis);

    /**
    // Find the bin number corresponding to the given cdf value in
    // an array which represents a cumulative distribution function
    // (the numbers in the array must not decrease).
    */
    template<typename Data>
    unsigned long quantileBinFromCdf(
        const Data* cdf, const unsigned long arrLen,
        const Data q)
    {
        if (!(arrLen > 1UL)) throw std::invalid_argument(
            "In ase::quantileBinFromCdf: insufficient amount of data");
        assert(cdf);

        if (q <= cdf[0])
        {
            unsigned long i = 1UL;
	    for (; cdf[i] == cdf[0] && i < arrLen; ++i);
	    return i - 1;
        }

        if (q >= cdf[arrLen - 1UL])
        {
            unsigned long i = arrLen - 1UL;
	    for (; cdf[i-1UL] == cdf[arrLen-1UL] && i>0; --i);
	    return i;
        }

        unsigned long imin = 0, imax = arrLen - 1UL;
        while (imax - imin > 1UL)
        {
            const unsigned long i = (imax + imin)/2UL;
            if (cdf[i] > q)
                imax = i;
            else if (cdf[i] < q)
                imin = i;
            else
            {
                for (imax = i; cdf[imax+1] == cdf[i]; ++imax);
                for (imin = i; cdf[imin-1] == cdf[i]; --imin);
                return (imin + imax)/2UL;
            }
        }
        return imin;
    }

    /** Conversion of moments about 0 to cumulants */
    void momentsToCumulants(const long double* moms, long double* cums,
                            unsigned order);

    /** Conversion of cumulants to moments about 0 */
    void cumulantsToMoments(const long double* cums, long double* moms,
                            unsigned order);
}

#endif // ASE_STATUTILS_HH_
