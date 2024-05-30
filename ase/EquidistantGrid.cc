#include <cmath>
#include <climits>
#include <stdexcept>

#include "ase/EquidistantGrid.hh"

namespace ase {
    EquidistantGrid::EquidistantGrid(const unsigned nCoords,
                                     const double xmin, const double xmax)
        : min_(xmin), max_(xmax), npt_(nCoords)
    {
        if (!(npt_ > 1U && npt_ < UINT_MAX/2U - 1U))
            throw std::invalid_argument("In ase::EquidistantGrid constructor: "
                                        "number of points is out of range");
        if (!(min_ < max_))
            throw std::invalid_argument("In ase::EquidistantGrid constructor: "
                                        "maximum must be larger than minimum");
        bw_ = (max_ - min_)/(npt_ - 1U);
    }                                 

    std::vector<double> EquidistantGrid::coords() const
    {
        std::vector<double> vec;
        vec.reserve(npt_);
        const unsigned nptm1 = npt_ - 1U;
        for (unsigned i=0; i<nptm1; ++i)
            vec.push_back(min_ + bw_*i);
        vec.push_back(max_);
        return vec;
    }

    double EquidistantGrid::coordinate(const unsigned i) const
    {
        if (i >= npt_) throw std::invalid_argument(
            "In ase::EquidistantGrid::coordinate: index out of range");
        if (i == npt_ - 1U)
            return max_;
        else
            return min_ + bw_*i;
    }

    std::pair<unsigned,double> EquidistantGrid::getInterval(const double x) const
    {
        if (x < min_ || x > max_) throw std::invalid_argument(
            "In ase::EquidistantGrid::getInterval: coordinate out of range");
        if (x == min_)
            return std::pair<unsigned,double>(0U, 1.0);
        else if (x == max_)
            return std::pair<unsigned,double>(npt_ - 2U, 0.0);
        else
        {
            unsigned binnum = static_cast<unsigned>(floor((x - min_)/bw_));
            if (binnum > npt_ - 2U)
                binnum = npt_ - 2U;
            double w = binnum + 1.0 - (x - min_)/bw_;
            if (w < 0.0)
                w = 0.0;
            else if (w > 1.0)
                w = 1.0;
            return std::pair<unsigned,double>(binnum, w);
        }
    }
}
