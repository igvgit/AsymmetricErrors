#ifndef ASE_OWENS_HH_
#define ASE_OWENS_HH_

namespace ase {
    namespace Private {
        void owen_values ( int &n_data, double &h, double &a, double &t );
        double q ( double h, double ah );
        double t ( double h, double a );
        void t_test ( );
        double tfun ( double h, double a, double ah );
    }
}

#endif // ASE_OWENS_HH_
