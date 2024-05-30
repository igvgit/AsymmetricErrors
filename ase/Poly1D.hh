#ifndef ASE_POLY1D_HH_
#define ASE_POLY1D_HH_

#include <vector>

namespace ase {
    class Poly1D
    {
    public:
        /*
        // Default constructor creates a polynomial which is
        // identically zero
        */
        inline Poly1D() : coeffs_(1, 0.0L) {}

        //@{
        /**
        // The length of the array of coefficients should be at least
        // degree+1. The highest degree coefficient is assumed to be
        // the last one in the "coeffs" array (0th degree coefficient
        // comes first).
        */
        Poly1D(const double* coeffs, unsigned degree);
        Poly1D(const long double* coeffs, unsigned degree);
        //@}

        //@{
        /**
        // The length of the vector of coefficients should be degree+1.
        // The highest degree coefficient is assumed to be the last one
        // in the vector (0th degree coefficient comes first).
        */
        explicit Poly1D(const std::vector<double>& c);
        explicit Poly1D(const std::vector<long double>& c);
        //@}

        /**
        // Note that the constructor from a floating point number
        // is not explicit. Implicit conversions from a floating
        // point number into a Poly1D are useful, for example, with
        // arithmetic operators.
        */
        inline Poly1D(const long double c0) : coeffs_(1, c0) {}

        /**
        // Construct a monomial of the given degree and with
        // the leading coefficient provided
        */
        Poly1D(unsigned degree, long double leadingCoeff);

        /**
        // At times it makes sense to reserve the space
        // for the polynomial coefficients (in particular,
        // with operator *=, when you use it more than once
        // and have some idea about the expected degree of
        // the result).
        */
        inline void reserve(const unsigned degree)
            {coeffs_.reserve(degree + 1U);}

        /**
        // Truncate the highest degree coefficients. The resulting
        // polynomial will be of degree "maxDegree" (or less, if the
        // new leading coefficient is 0).
        */
        void truncate(const unsigned maxDegree);

        /**
        // Explicitly truncate leading coefficients that are
        // zeros. Normally, the application code should not
        // use this function, as tracking and removal of leading
        // zeros happens automatically. It might still be useful
        // in certain difficult to anticipate underflow scenarios.
        */
        void truncateLeadingZeros();

        /** Set coefficient for a particular degree */
        void setCoefficient(unsigned degree, long double value);

        /** The degree of the polynomial */
        inline unsigned deg() const {return coeffs_.size() - 1U;}

        /** Return the coefficient for the given degree */
        inline long double operator[](const unsigned degree) const
            {return degree < coeffs_.size() ? coeffs_[degree] : 0.0L;}

        /** Return all coefficients */
        inline const std::vector<long double>& allCoefficients() const
            {return coeffs_;}

        /** The coefficient of the maximum degree monomial */
        long double leadingCoefficient() const;

        /** Number of real roots on the interval (a, b] */
        unsigned nRoots(long double a, long double b) const;

        /**
        // Real roots on the interval (a, b). The algorithm implemented
        // here is rather trivial and should not be used for high degree
        // polynomials. The number of roots is returned. Multiple roots
        // are counted only once. The length of the "roots" array should
        // be at least "deg()".
        */
        unsigned findRoots(long double a, long double b,
                           long double* roots) const;

        /** Polynomial value at x */
        long double operator()(long double x) const;

        /** Polynomial value and its derivative at x */
        void valueAndDerivative(long double x,
                                long double* value,
                                long double* derivative) const;

        /**
        // Check if all polynomial coefficients are within eps from
        // those of another polynomial. eps must be non-negative.
        */
        bool isClose(const Poly1D& r, long double eps) const;

        /** Check if the polynomial value is identical zero */
        inline bool isNull() const
            {return coeffs_.size() == 1U && coeffs_[0] == 0.0L;}

        /** Derivative polynomial */
        Poly1D derivative() const;

        /** Indefinite integral, with explicit additive constant */
        Poly1D integral(long double c) const;

        //@{
        /** Unary operator */
        inline Poly1D operator+() const {return *this;}
        Poly1D operator-() const;
        //@}

        //@{
        /** 
        // Binary operator on two polynomials or on a polynomial
        // and a floating point number
        */
        Poly1D operator*(const Poly1D&) const;
        Poly1D operator+(const Poly1D&) const;
        Poly1D operator-(const Poly1D&) const;
        //@}

        Poly1D& operator*=(const Poly1D&);
        Poly1D& operator+=(const Poly1D&);
        Poly1D& operator-=(const Poly1D&);

        /** Euclidean division of polynomials -- the ratio */
        Poly1D operator/(const Poly1D&) const;

        /** Euclidean division of polynomials -- the remainder */
        Poly1D operator%(const Poly1D&) const;

        inline bool operator==(const Poly1D& r) const
            {return coeffs_ == r.coeffs_;}
        inline bool operator!=(const Poly1D& r) const
            {return coeffs_ != r.coeffs_;}

        /** Monic polynomial of 0th degree (just 1.0) */
        inline static Poly1D monicDeg0() {return 1.0L;}

        /** Monic polynomial of degree 1, x + b */
        static Poly1D monicDeg1(long double b);

        /** Monic polynomial of degree 2, x^2 + b x + c */
        static Poly1D monicDeg2(long double b, long double c);

    private:
        long double singleRootOnInterval(long double a, long double fa,
                                         long double b, long double fb) const;
        std::vector<long double> coeffs_;
    };

    class Poly1DShifted
    {
    public:
        inline Poly1DShifted(const Poly1D& poly, const long double shift)
            : poly_(poly), shift_(shift) {}

        inline long double operator()(const long double x) const
            {return poly_(x - shift_);}

    private:
        Poly1D poly_;
        long double shift_;
    };
}

#endif // ASE_POLY1D_HH_
