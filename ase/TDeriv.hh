#ifndef ASE_TDERIV_HH_
#define ASE_TDERIV_HH_

namespace ase {
    namespace Private {
        template<class C>
        class TDeriv
        {
        public:
            inline TDeriv(const C& obj)
                : obj_(obj) {}
            
            inline typename C::value_type operator()(
                const typename C::value_type x) const
                {return obj_.derivative(x);}

        private:
            const C& obj_;
        };
    }
}

#endif // ASE_TDERIV_HH_
