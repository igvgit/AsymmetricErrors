#ifndef ASE_DERIVATIVEFUNCTORS_HH_
#define ASE_DERIVATIVEFUNCTORS_HH_

namespace ase {
    template<class T>
    class DerivativeFunctorHelper
    {
    public:
        inline DerivativeFunctorHelper(const T& t) : t_(t) {}

        inline double operator()(const double x) const
            {return t_.derivative(x);}
    private:
        const T& t_;
    };

    template<class T>
    class SecondDerivativeFunctorHelper
    {
    public:
        inline SecondDerivativeFunctorHelper(const T& t) : t_(t) {}

        inline double operator()(const double x) const
            {return t_.secondDerivative(x);}
    private:
        const T& t_;
    };

    template<class T>
    inline DerivativeFunctorHelper<T> DerivativeFunctor(const T& t)
        {return DerivativeFunctorHelper<T>(t);}

    template<class T>
    inline SecondDerivativeFunctorHelper<T> SecondDerivativeFunctor(const T& t)
        {return SecondDerivativeFunctorHelper<T>(t);}
}

#endif // ASE_DERIVATIVEFUNCTORS_HH_
