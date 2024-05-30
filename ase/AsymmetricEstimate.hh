#ifndef ASE_ASYMMETRICESTIMATE_HH_
#define ASE_ASYMMETRICESTIMATE_HH_

#include <string>
#include <iostream>
#include <vector>

#include "ase/Interval.hh"

namespace ase {
    enum ErrorType
    {
        P = 0,         // "pdf" or "systematic" asymmetric uncertainty
        L,             // "likelihood" asymmetric uncertainty
        N_ERROR_TYPES
    };

    class AsymmetricEstimate
    {
    public:
        //
        // For "likelihood" uncertainties, the uncertainties are
        // expected to be positive. The systematic uncertainties
        // can be both positive and negative.
        //
        AsymmetricEstimate(double centralValue, double sigmaPlus,
                           double sigmaMinus, ErrorType errorType);

        inline double location() const {return centralValue_;}
        inline double sigmaPlus() const {return sigPlus_;}
        inline double sigmaMinus() const {return sigMinus_;}
        inline ErrorType errorType() const {return errorType_;}

        inline Interval<double> intervalEstimate() const
        {
            return Interval<double>(centralValue_ - sigMinus_,
                                    centralValue_ + sigPlus_,
                                    CLOSED_INTERVAL);
        }

        // The following methods will throw an exception
        // in case the uncertainties have opposite signs
        double width() const;
        double asymmetry() const;

    private:
        void validate() const;

        double centralValue_;
        double sigPlus_;
        double sigMinus_;
        ErrorType errorType_;
    };

    // Parse the estimate represented by one line of ASCII text
    AsymmetricEstimate parseAsymmetricEstimate(const std::string& line);

    // Read a file containing asymmetric estimates, one per line
    std::vector<AsymmetricEstimate> readAsymmetricEstimates(
        const std::string& filename);
}

// Dump a text reprsesentation of the estimate
std::ostream& operator<<(std::ostream& os, const ase::AsymmetricEstimate& e);

bool operator==(const ase::AsymmetricEstimate& l,
                const ase::AsymmetricEstimate& r);

inline bool operator!=(const ase::AsymmetricEstimate& l,
                       const ase::AsymmetricEstimate& r)
{return !(l == r);}

#endif // ASE_ASYMMETRICESTIMATE_HH_
