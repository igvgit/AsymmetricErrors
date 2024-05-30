#ifndef ASE_EQUIDISTANTGRID_HH_
#define ASE_EQUIDISTANTGRID_HH_

#include <vector>
#include <utility>

namespace ase {
    /** Class that helps with data provided for equidistant points */
    class EquidistantGrid
    {
    public:
        // The number of coordinates must be at least 2
        EquidistantGrid(unsigned nCoords, double min, double max);

        // Basic accessors
        inline unsigned nCoords() const {return npt_;}
        inline double min() const {return min_;}
        inline double max() const {return max_;}

        // The following function returns the grid interval number and
        // the weight of the point at the left side of the interval.
        // The weight will be set to 1 if the given coordinate coincides
        // with the grid point and will decay to 0 linearly as the
        // coordinate moves towards the next point on the right.
        //
        // Calls with coordinates below min or above max will throw
        // std::invalid_argument.
        std::pair<unsigned,double> getInterval(double coordinate) const;

        // Convenience methods
        std::vector<double> coords() const;
        double coordinate(unsigned i) const;
        inline double length() const {return max_ - min_;}
        inline bool isUniform() const {return true;}
        inline unsigned nIntervals() const {return npt_ - 1;}
        inline double intervalWidth() const {return bw_;}
        inline double intervalWidth(unsigned) const {return bw_;}

    private:
        double min_;
        double max_;
        double bw_;
        unsigned npt_;
    };
}

#endif // ASE_EQUIDISTANTGRID_HH_
