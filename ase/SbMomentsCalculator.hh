#ifndef ASE_SBMOMENTSCALCULATOR_HH_
#define ASE_SBMOMENTSCALCULATOR_HH_

//=========================================================================
// SbMomentsCalculator.hh
//
// Internal utility classes used for calculating transformation
// parameters of Johnson's S_b distribution. Applications codes
// should not use any of these classes explicitly, as they might
// change in the future.
//
// Author: I. Volobouev
//
// May 2010, June 2023
//=========================================================================

#include "ase/GaussHermiteQuadrature.hh"
#include "ase/GaussLegendreQuadrature.hh"

namespace ase {
    struct SbMomentsCalculator
    {
        inline virtual ~SbMomentsCalculator() {}

        // The following function should return "true" if the
        // arguments (and the answer) are reasonable
        virtual bool calculate(
            long double p0, long double p1,
            long double *mean, long double *var,
            long double *skew, long double *kurt,
            long double *dskewdp0, long double *dskewdp1,
            long double *dkurtdp0, long double *dkurtdp1) const = 0;

        virtual void getParameters(
            long double gamma, long double delta,
            long double *p0, long double *p1) const = 0;

        virtual void getGammaDelta(
            long double p0, long double p1,
            long double *gamma, long double *delta) const = 0;
    };

    struct SbMomentsBy6Integrals : public SbMomentsCalculator
    {
        inline virtual ~SbMomentsBy6Integrals() {}

        // The "results" array must have at least 6 elements
        virtual bool integrate6(
            long double a, long double b, long double *results) const = 0;

        bool calculate(
            long double p0, long double p1,
            long double *mean, long double *var,
            long double *skew, long double *kurt,
            long double *dskewdp0, long double *dskewdp1,
            long double *dkurtdp0, long double *dkurtdp1) const;

        void getParameters(
            long double gamma, long double delta,
            long double *p0, long double *p1) const;

        void getGammaDelta(
            long double p0, long double p1,
            long double *gamma, long double *delta) const;
    };

    struct SbMomentsGaussHermite : public SbMomentsBy6Integrals
    {
        // The number of points in the constructor below should be
        // supported by the "GaussHermiteQuadrature" integrator:
        // see "GaussHermiteQuadrature.hh" header file for more info
        explicit SbMomentsGaussHermite(unsigned npoints=256U);
        inline virtual ~SbMomentsGaussHermite() {}

        bool integrate6(
            long double a, long double b, long double *results) const;
    private:
        GaussHermiteQuadrature quad_;
    };

    struct SbMomentsMultiZone : public SbMomentsBy6Integrals
    {
        // The number of points in the constructor below should be
        // supported by the "GaussLegendreQuadrature" integrator:
        // see "GaussLegendreQuadrature.hh" header file for more info
        explicit SbMomentsMultiZone(unsigned npoints=128U);
        inline virtual ~SbMomentsMultiZone() {}

        bool integrate6(
            long double a, long double b, long double *results) const;
    private:
        void central_integ_big_a(long double a, long double b,
                                 long double* p) const;
        GaussLegendreQuadrature quad_;
    };

    struct SbMomentsMix : public SbMomentsBy6Integrals
    {
        explicit SbMomentsMix(double mixPoint=0.95, unsigned npointsHemite=256U,
                              unsigned npointsLegendre=128U);
        inline virtual ~SbMomentsMix() {}

        inline double mixPoint() const {return mixPoint_;}
        inline void setMixPoint(double mixPoint) {mixPoint_ = mixPoint;}

        bool integrate6(
            long double a, long double b, long double *results) const;
    private:
        SbMomentsGaussHermite gh;
        SbMomentsMultiZone mz;
        double mixPoint_;
    };

    class SbMomentsFunctor
    {
    public:
        SbMomentsFunctor(long double a, long double b, unsigned n);
        inline virtual ~SbMomentsFunctor() {}
        long double operator()(const long double& x) const;

    private:
        SbMomentsFunctor();
        long double a_;
        long double b_;
        unsigned n_;
    };

    class SbMomentsGaussFunctor
    {
    public:
        SbMomentsGaussFunctor(long double a, long double b, unsigned n);
        inline virtual ~SbMomentsGaussFunctor() {}
        long double operator()(const long double& x) const;

    private:
        SbMomentsGaussFunctor();
        long double a_;
        long double b_;
        unsigned n_;
    };

    class SbMomentsInvErfFunctor
    {
    public:
        SbMomentsInvErfFunctor(long double a, long double b, unsigned n);
        inline virtual ~SbMomentsInvErfFunctor() {}
        long double operator()(const long double& x) const;

    private:
        SbMomentsInvErfFunctor();
        long double a_;
        long double b_;
        unsigned n_;
    };

    struct SbMoments0SkewBigKurt : public SbMomentsCalculator
    {
        inline virtual ~SbMoments0SkewBigKurt() {}

        bool calculate(
            long double p0, long double p1,
            long double *mean, long double *var,
            long double *skew, long double *kurt,
            long double *dskewdp0, long double *dskewdp1,
            long double *dkurtdp0, long double *dkurtdp1) const;

        void getParameters(
            long double gamma, long double delta,
            long double *p0, long double *p1) const;

        void getGammaDelta(
            long double p0, long double p1,
            long double *gamma, long double *delta) const;
    };

    struct SbMomentsBigDelta : public SbMomentsCalculator
    {
        inline virtual ~SbMomentsBigDelta() {}

        bool calculate(
            long double p0, long double p1,
            long double *mean, long double *var,
            long double *skew, long double *kurt,
            long double *dskewdp0, long double *dskewdp1,
            long double *dkurtdp0, long double *dkurtdp1) const;

        void getParameters(
            long double gamma, long double delta,
            long double *p0, long double *p1) const;

        void getGammaDelta(
            long double p0, long double p1,
            long double *gamma, long double *delta) const;
    };

    struct SbMomentsBigGamma : public SbMomentsCalculator
    {
        explicit SbMomentsBigGamma(unsigned maxdeg=20U);
        inline virtual ~SbMomentsBigGamma() {}

        void setMaxDeg(unsigned maxdeg);
        inline unsigned maxdeg() const {return maxdeg_;}
        unsigned bestDeg(long double p0, long double p1) const;

        bool calculate(
            long double p0, long double p1,
            long double *mean, long double *var,
            long double *skew, long double *kurt,
            long double *dskewdp0, long double *dskewdp1,
            long double *dkurtdp0, long double *dkurtdp1) const;

        void getParameters(
            long double gamma, long double delta,
            long double *p0, long double *p1) const;

        void getGammaDelta(
            long double p0, long double p1,
            long double *gamma, long double *delta) const;
    private:
        unsigned maxdeg_;
    };
}

#endif // ASE_SBMOMENTSCALCULATOR_HH_
