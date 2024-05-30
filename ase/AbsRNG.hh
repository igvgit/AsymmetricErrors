#ifndef ASE_ABSRNG_HH_
#define ASE_ABSRNG_HH_

namespace ase {
    /** Base class for 1-d random number generators */
    struct AbsRNG
    {
        inline virtual ~AbsRNG() {}

        /** Generate random numbers on (0, 1) */
        virtual double operator()() = 0;
    };
}

#endif // ASE_ABSRNG_HH_
