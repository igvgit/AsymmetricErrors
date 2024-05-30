#ifndef ASE_INTERVAL_HH_
#define ASE_INTERVAL_HH_

#include <cmath>
#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace ase {
    enum IntervalType
    {
        OPEN_INTERVAL = 0,
        CLOSED_INTERVAL,
        LEFT_CLOSED_INTERVAL,
        RIGHT_CLOSED_INTERVAL,
        RIGHT_OPEN_INTERVAL = LEFT_CLOSED_INTERVAL,
        LEFT_OPEN_INTERVAL = RIGHT_CLOSED_INTERVAL
    };

    /** A few useful operations with 1-d intervals */
    template <typename Numeric>
    class Interval
    {
    public:
        // Default constructor makes an empty interval
        inline Interval() : min_(), max_(), t_(OPEN_INTERVAL) {}

        inline Interval(const Numeric x1, const Numeric x2, const IntervalType t)
            : min_(x1 < x2 ? x1 : x2), max_(x1 < x2 ? x2 : x1), t_(t) {}

        inline Numeric min() const {return min_;}
        inline Numeric max() const {return max_;}
        inline IntervalType type() const {return t_;}
        inline Numeric length() const {return max_ - min_;}
        inline Numeric midpoint() const {return (max_ + min_)/2;}
        inline bool empty() const
            {return min_ == max_ && t_ != CLOSED_INTERVAL;}
        inline bool includesLeft() const
            {return t_ == CLOSED_INTERVAL || t_ == LEFT_CLOSED_INTERVAL;}
        inline bool includesRight() const
            {return t_ == CLOSED_INTERVAL || t_ == RIGHT_CLOSED_INTERVAL;}

        inline bool contains(const Numeric x) const
        {
            switch (t_)
            {
            case OPEN_INTERVAL:
                return min_ < x && x < max_;
            case CLOSED_INTERVAL:
                return min_ <= x && x <= max_;
            case LEFT_CLOSED_INTERVAL:
                return min_ <= x && x < max_;
            case RIGHT_CLOSED_INTERVAL:
                return min_ < x && x <= max_;
            default:
                assert(!"Unhandled case in ase::Interval::contains");
                return false;
            }
        }

        inline bool contains(const Interval& r) const
        {
            if (r.empty())
                return true;
            if (empty())
                return false;
            const bool rOpen = r.t_ == RIGHT_OPEN_INTERVAL || r.t_ == OPEN_INTERVAL;
            const bool lOpen = r.t_ == LEFT_OPEN_INTERVAL || r.t_ == OPEN_INTERVAL;
            const bool lContained = lOpen ? min_ <= r.min_ && r.min_ <= max_ :
                                            contains(r.min_);
            const bool rContained = rOpen ? min_ <= r.max_ && r.max_ <= max_ :
                                            contains(r.max_);
            return lContained && rContained;
        }

        inline Numeric distance(const Numeric x) const
        {
            if (empty()) throw std::runtime_error(
                "In ase::Interval::distance: "
                "distance to an empty interval is undefined");
            if (min_ <= x && x <= max_)
                return 0.0;
            else
                return std::min(std::abs(x - min_), std::abs(x - max_));
        }

        inline Interval overlap(const Interval& r) const
        {
            if (empty() || r.empty())
                return Interval();
            else
            {
                Interval over;
                if (max_ == r.min_)
                {
                    over.min_ = r.min_;
                    over.max_ = max_;
                    over.t_ = fromInclusions(r.includesLeft(), includesRight());
                }
                else if (r.max_ == min_)
                {
                    over.min_ = min_;
                    over.max_ = r.max_;
                    over.t_ = fromInclusions(includesLeft(), r.includesRight());
                }
                else if (max_ > r.min_ && r.max_ > min_)
                {
                    over.min_ = min_ < r.min_ ? r.min_ : min_;
                    over.max_ = max_ < r.max_ ? max_ : r.max_;
                    bool includeLeft = min_ < r.min_ ? r.includesLeft() : includesLeft();
                    bool includeRight = max_ < r.max_ ? includesRight() : r.includesRight();
                    if (min_ == r.min_)
                        includeLeft = r.includesLeft() && includesLeft();
                    if (max_ == r.max_)
                        includeRight = includesRight() && r.includesRight();
                    over.t_ = fromInclusions(includeLeft, includeRight);
                }
                return over;
            }
        }

        static inline IntervalType fromInclusions(
            const bool includeLeft, const bool includeRight)
        {
            if (includeLeft)
            {
                if (includeRight)
                    return CLOSED_INTERVAL;
                else
                    return LEFT_CLOSED_INTERVAL;
            }
            else
            {
                if (includeRight)
                    return RIGHT_CLOSED_INTERVAL;
                else
                    return OPEN_INTERVAL;
            }
        }

    private:
        Numeric min_;
        Numeric max_;
        IntervalType t_;
    };
}

#endif // ASE_INTERVAL_HH_
