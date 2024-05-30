#include <string>
#include <sstream>

#include "ase/miscUtils.hh"

namespace ase {
    unsigned printed_unsigned_width(const unsigned u)
    {
        std::ostringstream os;
        os << u;
        const std::string& st = os.str();
        return st.size();
    }
}
