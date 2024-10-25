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

        // Unary operators
        AsymmetricEstimate operator-() const;
        inline AsymmetricEstimate operator+() const
        {
            return *this;
        }

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

// Comparison for equality
bool operator==(const ase::AsymmetricEstimate& l,
                const ase::AsymmetricEstimate& r);

inline bool operator!=(const ase::AsymmetricEstimate& l,
                       const ase::AsymmetricEstimate& r)
{
    return !(l == r);
}

// Convenience binary arithmetic operations with exact numbers
ase::AsymmetricEstimate operator*(const ase::AsymmetricEstimate& l,
                                  const double& r);

inline ase::AsymmetricEstimate operator*(const double& l,
                                         const ase::AsymmetricEstimate& r)
{
    return r * l;
}

ase::AsymmetricEstimate operator/(const ase::AsymmetricEstimate& l,
                                  const double& r);
    
inline ase::AsymmetricEstimate operator+(const ase::AsymmetricEstimate& l,
                                         const double& r)
{
    return ase::AsymmetricEstimate(l.location() + r, l.sigmaPlus(),
                                   l.sigmaMinus(), l.errorType());
}

inline ase::AsymmetricEstimate operator+(const double& l,
                                         const ase::AsymmetricEstimate& r)
{
    return r + l;
}

inline ase::AsymmetricEstimate operator-(const ase::AsymmetricEstimate& l,
                                         const double& r)
{
    return l + (-1.0*r);
}

inline ase::AsymmetricEstimate operator-(const double& l,
                                         const ase::AsymmetricEstimate& r)
{
    return r*(-1.0) + l;
}

#endif // ASE_ASYMMETRICESTIMATE_HH_
