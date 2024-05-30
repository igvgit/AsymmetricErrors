#ifndef ASE_LEGENDREDISTRO1D_HH_
#define ASE_LEGENDREDISTRO1D_HH_

#include <vector>

#include "ase/AbsLocationScaleFamily.hh"
#include "ase/LegendreOrthoPoly1D.hh"

namespace ase {
    /** 
    // Distribution whose density is represented by series in
    // _orthonormal_ Legendre polynomials
    */
    class LegendreDistro1D : public AbsLocationScaleFamily
    {
    public:
        static const bool isFullOPAT = false;

        /**
        // The series coefficients start with degree 1 (note, not 0).
        // The coefficient for degree 0 is 1/2 (fixed by normalization).
        // The coefficients are given for the expansion on the [-1, 1]
        // interval, and then the density is shifted and scaled in the
        // usual manner.
        //
        // If "ensurePositivity" is true, the code will throw
        // std::invalid_argument in case the density becomes 0 or negative
        // on the [-1, 1] interval. If this argument is false, the
        // code will not throw for this reason. Instead, one should call
        // the method "isNonNegative()" to figure out whether a valid
        // distribution was constructed.
        */
        LegendreDistro1D(double location, double scale,
                         const std::vector<double>& coeffs,
                         bool ensurePositivity = true);

        inline virtual LegendreDistro1D* clone() const override
            {return new LegendreDistro1D(*this);}

        inline virtual ~LegendreDistro1D() override {}

        inline unsigned nCoeffs() const {return allCoeffs_.size() - 1U;}
        inline double getCoeff(const unsigned i) const
            {return allCoeffs_.at(i + 1U);}

        inline virtual bool isNonNegative() const override
            {return isPositive_;}

        virtual bool isUnimodal() const override;

        inline virtual std::string classname() const override
            {return "LegendreDistro1D";}

    private:
        void checkPositive(bool validatePositivity);
        void setupCoeffs(const double* coeffs, unsigned maxdeg);
        double calculateMoment(double mu, unsigned power) const;
        void calculateCumulants() const;

        virtual double unscaledDensity(double x) const override;
        virtual double unscaledDensityDerivative(double x) const override;
        virtual double unscaledCdf(double x) const override;
        virtual double unscaledExceedance(double x) const override;
        virtual double unscaledQuantile(double x) const override;
        virtual double unscaledInvExceedance(double x) const override;
        virtual double unscaledCumulant(unsigned n) const override;
        virtual double unscaledDescentDelta(
            bool isToTheRight, double deltaLnL) const override;
        virtual double unscaledMode() const override;

        LegendreOrthoPoly1D poly_;
        std::vector<double> allCoeffs_;
        std::vector<double> integCoeffs_;
        std::vector<double> derivCoeffs_;
        bool isPositive_;

        mutable double cumulants_[5];
        mutable bool cumulantsCalculated_;
    };
}

#endif // ASE_LEGENDREDISTRO1D_HH_
