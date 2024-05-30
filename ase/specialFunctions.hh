#ifndef ASE_SPECIALFUNCTIONS_HH_
#define ASE_SPECIALFUNCTIONS_HH_

namespace ase {
    /** Inverse cumulative distribition function for 1-d Gaussian */
    double inverseGaussCdf(double cdf);

    /** Owen's T function */
    double owensT(double h, double alpha);

    /** The gamma function for positive real arguments */
    double Gamma(double x);

    /** Incomplete gamma ratio */
    double incompleteGamma(double a, double x);

    /** Inverse incomplete gamma ratio */
    double inverseIncompleteGamma(double a, double x);

    /** Incomplete gamma ratio complement */
    double incompleteGammaC(double a, double x);

    /** Inverse incomplete gamma ratio complement */
    double inverseIncompleteGammaC(double a, double x);
}

#endif // ASE_SPECIALFUNCTIONS_HH_
