#ifndef ASE_GAUSSHERMITEQUADRATURE_HH_
#define ASE_GAUSSHERMITEQUADRATURE_HH_

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace ase {
    /** 
    // Gauss-Hermite quadrature. Internally, operations are performed
    // in long double precision.
    */
    class GaussHermiteQuadrature
    {
    public:
        /**
        // At the moment, the following numbers of points are supported:
        // 4, 8, 16, 32, 64, 100, 128, 256, 512.
        //
        // If an unsupported number of points is given in the
        // constructor, std::invalid_argument exception will be thrown.
        //
        // Note that, for the quadrature with 512 points, the
        // arguments become as large as 31.4 and the weights become
        // as small as 4.9e-430. Use of such weights with doubles is
        // rather meaningless. Long double integrands are needed.
        */
        explicit GaussHermiteQuadrature(unsigned npoints);

        /** Return the number of quadrature points */
        inline unsigned npoints() const {return npoints_;}

        /** Perform the quadrature */
        template <class Functor>
        long double integrate(const Functor& function) const
        {
            std::pair<long double, long double>* results = &buf_[0];
            const unsigned halfpoints = npoints_/2;
            for (unsigned i=0; i<halfpoints; ++i)
            {
                long double a = a_[i];
                long double v = w_[i]*function(a);
                results[2*i].first = std::abs(v);
                results[2*i].second = v;
                a = -a_[i];
                v = w_[i]*function(a);
                results[2*i+1].first = std::abs(v);
                results[2*i+1].second = v;
            }
            std::sort(results, results+npoints_);
            long double sum = 0.0L;
            for (unsigned i=0; i<npoints_; ++i)
                sum += results[i].second;
            return sum;
        }

        /** Perform the quadrature with Gaussian density weight */
        template <class Functor>
        long double integrateProb(const long double mean,
                                  const long double sigma,
                                  const Functor& function) const
        {
            const long double sqr2 = 1.41421356237309504880168872421L;
            const long double sqrpi = 1.77245385090551602729816748334L;

            if (sigma <= 0.0L) throw std::invalid_argument(
                "In ase::GaussHermiteQuadrature::integrateProb: "
                "sigma must be positive");
            std::pair<long double, long double>* results = &buf_[0];
            const unsigned halfpoints = npoints_/2;
            for (unsigned i=0; i<halfpoints; ++i)
            {
                const long double delta = sqr2*sigma*a_[i];
                long double a = mean + delta;
                long double v = w_[i]*function(a);
                results[2*i].first = std::abs(v);
                results[2*i].second = v;
                a = mean - delta;
                v = w_[i]*function(a);
                results[2*i+1].first = std::abs(v);
                results[2*i+1].second = v;
            }
            std::sort(results, results+npoints_);
            long double sum = 0.0L;
            for (unsigned i=0; i<npoints_; ++i)
                sum += results[i].second;
            return sum/sqrpi;
        }

        /** Return abscissae for all points */
        void getAllAbscissae(long double* abscissae, unsigned len) const;

        /** Return weights for all points */
        void getAllWeights(long double* weights, unsigned len) const;

        /** Check if the rule with the given number of points is supported */
        static bool isAllowed(unsigned npoints);

        /** The complete set of allowed rules, in the increasing order */
        static std::vector<unsigned> allowedNPonts();

        /**
        // Minimum number of points, among the supported rules, which
        // integrates a polynomial with the given degree exactly (with
        // the appropriate weight). Returns 0 if the degree is out of range.
        */
        static unsigned minimalExactRule(unsigned polyDegree);

    private:
        const long double* a_;
        const long double* w_;
        // The following can potentially break multithreaded code.
        // Do not use the same quadrature object in different threads.
        mutable std::vector<std::pair<long double, long double> > buf_;
        unsigned npoints_;

#ifdef SWIG
    public:
        template <class Functor>
        inline double integrate2(const Functor& function) const
        {
            return integrate(function);
        }

        template <class Functor>
        inline double integrateProb2(const double mean,
                                     const double sigma,
                                     const Functor& function) const
        {
            return integrateProb(mean, sigma, function);
        }
#endif // SWIG
    };
}

#endif // ASE_GAUSSHERMITEQUADRATURE_HH_
