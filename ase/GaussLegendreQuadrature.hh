#ifndef ASE_GAUSSLEGENDREQUADRATURE_HH_
#define ASE_GAUSSLEGENDREQUADRATURE_HH_

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace ase {
    /** 
    // Gauss-Legendre quadrature. Internally, operations are performed
    // in long double precision.
    */
    class GaussLegendreQuadrature
    {
    public:
        /**
        // At the moment, the following numbers of points are supported:
        // 2, 4, 6, 8, 10, 12, 16, 32, 64, 100, 128, 256, 512, 1024.
        //
        // If an unsupported number of points is given in the
        // constructor, std::invalid_argument exception will be thrown.
        */
        explicit GaussLegendreQuadrature(unsigned npoints);

        inline virtual ~GaussLegendreQuadrature() {}

        /** Return the number of quadrature points */
        inline unsigned npoints() const {return npoints_;}

        /** Perform the quadrature on the [left, right] interval */
        template <class Functor>
        long double integrate(const Functor& function,
                              const long double left,
                              const long double right) const
        {
            std::pair<long double, long double>* results = &buf_[0];
            const unsigned halfpoints = npoints_/2;
            const long double midpoint = (left + right)/2.0L;
            const long double unit = (right - left)/2.0L;
            for (unsigned i=0; i<halfpoints; ++i)
            {
                long double a = midpoint + unit*a_[i];
                long double v = w_[i]*function(a);
                results[2*i].first = std::abs(v);
                results[2*i].second = v;
                a = midpoint - unit*a_[i];
                v = w_[i]*function(a);
                results[2*i+1].first = std::abs(v);
                results[2*i+1].second = v;
            }
            if (npoints_ > 2U)
                std::sort(results, results+npoints_);
            long double sum = 0.0L;
            for (unsigned i=0; i<npoints_; ++i)
                sum += results[i].second;
            return sum*unit;
        }

        /**
        // This method splits the interval [left, right] into "nsplit"
        // subintervals of equal length, applies Gauss-Legendre
        // quadrature to each subinterval, and sums the results.
        */
        template <class Functor>
        long double integrate(const Functor& function,
                              const long double left, const long double right,
                              const unsigned nsplit) const
        {
            if (!nsplit) throw std::invalid_argument(
                "In ase::GaussLegendreQuadrature::integrate: "
                "number of subintervals must be positive");
            if (nsplit == 1U)
                return integrate(function, left, right);
            else
            {
                std::vector<std::pair<long double, long double> > buf(nsplit);
                const long double step = (right - left)/nsplit;
                long double b = left;
                for (unsigned i=0; i<nsplit; ++i)
                {
                    const long double a = b;
                    b = (i == nsplit - 1U ? right : a + step);
                    const long double v = integrate(function, a, b);
                    buf[i].first = std::abs(v);
                    buf[i].second = v;
                }
                std::sort(buf.begin(), buf.end());
                long double acc = 0.0L;
                for (unsigned i=0; i<nsplit; ++i)
                    acc += buf[i].second;
                return acc;
            }
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
        // integrates a polynomial with the given degree exactly.
        // Returns 0 if the degree is out of range.
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
        inline double integrate2(const Functor& function,
                                 const double left, const double right) const
        {
            return integrate(function, left, right);
        }

        template <class Functor>
        inline double integrate2(const Functor& function,
                                 const double left, const double right,
                                 const unsigned nsplit) const
        {
            return integrate(function, left, right, nsplit);
        }
#endif // SWIG
    };
}

#endif // ASE_GAUSSLEGENDREQUADRATURE_HH_
