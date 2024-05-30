#ifndef ASE_CRANDOMGEN_HH_
#define ASE_CRANDOMGEN_HH_

#include "ase/AbsRNG.hh"

namespace ase {
    /**
    // Wrapper for functions which look like "double generate()",
    // for example, "drand48" from cstdlib.
    */
    class CRandomGen : public AbsRNG
    {
    public:
        inline explicit CRandomGen(double (*fcn)()) : f_(fcn) {}
        inline virtual ~CRandomGen() override {}

        inline double operator()() override {return f_();}

    private:
        double (*f_)();
    };
}

#endif // ASE_CRANDOMGEN_HH_
