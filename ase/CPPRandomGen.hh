#ifndef ASE_CPPRANDOMGEN_HH_
#define ASE_CPPRANDOMGEN_HH_

#include <random>

#include "ase/AbsRNG.hh"

namespace ase {
    /**
    // Wrapper class for random number generators defined in the C++11
    // standard. The template type should be one of the random engines
    // defined by C++11. Example:
    //
    // std::random_device rd;
    // std::mt19937 eng(rd());
    // ase::CPPRandomGen<decltype(eng)> gen(eng);
    //
    // Now, gen() will produce pseudo-random numbers. Note that this
    // class holds only a reference to the engine but not the engine
    // itself.
    */
    template<class RandomEngine>
    class CPPRandomGen : public AbsRNG
    {
    public:
        inline explicit CPPRandomGen(RandomEngine& fcn)
            : f_(fcn), uni_(0.0, 1.0) {}

        inline virtual ~CPPRandomGen() override {}

        inline double operator()() override
            {return uni_(f_);}

    private:
        RandomEngine& f_;
        std::uniform_real_distribution<double> uni_;
    };
}

#endif // ASE_CPPRANDOMGEN_HH_
