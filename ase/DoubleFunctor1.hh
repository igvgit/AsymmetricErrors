#ifndef ASE_DOUBLEFUNCTOR1_HH_
#define ASE_DOUBLEFUNCTOR1_HH_

namespace ase {
    /** Functor wrapper for functions which look like "double fcn(double x)" */
    class DoubleFunctor1
    {
    public:
        inline explicit DoubleFunctor1(double (*fcn)(double)) : f_(fcn) {}

        inline double operator()(const double x) const {return f_(x);}

    private:
        double (*f_)(double);
    };

    /**
    // Functor wrapper for functions which look like
    // "long double fcn(long double x)"
    */
    class LongDoubleFunctor1
    {
    public:
        inline explicit LongDoubleFunctor1(long double (*fcn)(long double)) : f_(fcn) {}

        inline long double operator()(const long double x) const {return f_(x);}

    private:
        long double (*f_)(long double);
    };
}

#endif // ASE_DOUBLEFUNCTOR1_HH_
