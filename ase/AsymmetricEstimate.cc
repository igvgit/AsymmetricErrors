#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ase/AsymmetricEstimate.hh"

namespace ase {
    static const char* errorTags[N_ERROR_TYPES] = {"P", "L"};

    AsymmetricEstimate::AsymmetricEstimate(
        const double i_centralValue,
        const double i_positiveUncertainty,
        const double i_negativeUncertainty,
        const ErrorType i_errorType)
        : centralValue_(i_centralValue),
          sigPlus_(i_positiveUncertainty),
          sigMinus_(i_negativeUncertainty),
          errorType_(i_errorType)
    {
        validate();
    }

    void AsymmetricEstimate::validate() const
    {
        switch (errorType_)
        {
        case ErrorType::P:
            break;

        case ErrorType::L:
            if (sigPlus_ < 0.0) throw std::invalid_argument(
                "In ase::AsymmetricEstimate::validate: "
                "sigPlus must not be negative for 'L' errors");
            if (sigMinus_ < 0.0) throw std::invalid_argument(
                "In ase::AsymmetricEstimate::validate: "
                "sigMinus must not be negative for 'L' errors");
            break;

        default:
            throw std::runtime_error(
                "In ase::AsymmetricEstimate::validate: "
                "unhandled switch case. This is a bug. Please report.");
            break;
        }
    }

    double AsymmetricEstimate::width() const
    {
        if (sigPlus_*sigMinus_ < 0.0) throw std::runtime_error(
            "In ase::AsymmetricEstimate::width: uncertainties "
            "have opposite signs");
        return (std::abs(sigPlus_) + std::abs(sigMinus_))/2.0;
    }

    double AsymmetricEstimate::asymmetry() const
    {
        if (sigPlus_*sigMinus_ < 0.0) throw std::runtime_error(
            "In ase::AsymmetricEstimate::asymmetry: uncertainties "
            "have opposite signs");
        const double sp = std::abs(sigPlus_);
        const double sm = std::abs(sigMinus_);
        const double ssum = sp + sm;
        if (!ssum) throw std::runtime_error(
            "In ase::AsymmetricEstimate::asymmetry: asymmetry is "
            "undefined if both uncertainties are 0");
        const double as = (sp - sm)/ssum;
        return sigPlus_ >= 0.0 || sigMinus_ >= 0.0 ? as : -as;
    }

    AsymmetricEstimate AsymmetricEstimate::operator-() const
    {
        return *this * (-1.0);
    }

    AsymmetricEstimate parseAsymmetricEstimate(const std::string& line)
    {
        std::istringstream is;
        is.str(line);
        double cv, pu, nu;
        ErrorType eType = ErrorType::N_ERROR_TYPES;
        std::string eTypeStr;

        is >> eTypeStr;
        if (eTypeStr.size() == 1U)
        {
            if (eTypeStr[0] == 'P' || eTypeStr[0] == 'p')
                eType = ErrorType::P;
            if (eTypeStr[0] == 'L' || eTypeStr[0] == 'l')
                eType = ErrorType::L;
        }
        if (is.fail() || eType == ErrorType::N_ERROR_TYPES)
            throw std::invalid_argument("column 1 is invalid or not present");

        is >> cv;
        if (is.fail())
            throw std::invalid_argument("column 2 is invalid or not present");

        is >> pu;
        if (is.fail())
            throw std::invalid_argument("column 3 is invalid or not present");

        is >> nu;
        if (is.fail())
            throw std::invalid_argument("column 4 is invalid or not present");

        if (is)
        {
            std::string dummy;
            is >> dummy;
            if (!dummy.empty())
                throw std::invalid_argument("found extra characters after column 4");
        }

        return AsymmetricEstimate(cv, pu, nu, eType);
    }

    std::vector<AsymmetricEstimate> readAsymmetricEstimates(
        const std::string& filename)
    {
        std::ifstream ifs(filename);
        if (!ifs.is_open())
        {
            std::ostringstream os;
            os << "In ase::readAsymmetricEstimates: "
               << "failed to open file \"" << filename << "\"";
            throw std::invalid_argument(os.str());
        }

        std::vector<AsymmetricEstimate> result;
        std::string linebuf;
        unsigned long lineNumber = 0;

        while (ifs)
        {
            std::getline(ifs, linebuf);
            ++lineNumber;
            const unsigned long len = linebuf.size();
            if (len == 0UL)
                continue;

            // Ignore lines which are pure white space
            // or which start with an arbitrary number
            // of white space characters followed by #.
            bool isComment = false;
            bool allSpace = true;
            char* line = &linebuf[0];
            for (unsigned long i=0; i<len; ++i)
            {
                // Convert commas into white space
                if (line[i] == ',')
                    line[i] = ' ';
                if (isspace(line[i]))
                    continue;
                if (allSpace && line[i] == '#')
                {
                    isComment = true;
                    break;
                }
                allSpace = false;
            }
            if (isComment || allSpace)
                continue;

            // Read the data into the buffer
            try {
                const AsymmetricEstimate& est =
                    parseAsymmetricEstimate(linebuf);
                result.push_back(est);
            }
            catch (const std::invalid_argument& e) {
                std::ostringstream os;
                os << "In ase::readAsymmetricEstimates: "
                   << "failed to parse line " << lineNumber
                   << " in file \"" << filename << "\": " << e.what();
                throw std::invalid_argument(os.str());
            }
        }

        if (result.empty())
        {
            std::ostringstream os;
            os << "In ase::readAsymmetricEstimates: "
               << "no asymmetric estimate specifications found "
               << "in file \"" << filename << "\"";
            throw std::invalid_argument(os.str());
        }

        return result;
    }
}

std::ostream& operator<<(std::ostream& os, const ase::AsymmetricEstimate& e)
{
    // Save flags and restore them after we are done
    const ase::ErrorType eType = e.errorType();
    assert(eType < ase::ErrorType::N_ERROR_TYPES);
    std::ios_base::fmtflags fl(os.flags());
    os.precision(16);
    os << ase::errorTags[eType]
       << ' ' << e.location()
       << ' ' << e.sigmaPlus()
       << ' ' << e.sigmaMinus();
    os.flags(fl);
    return os;
}

bool operator==(const ase::AsymmetricEstimate& l,
                const ase::AsymmetricEstimate& r)
{
    return l.errorType() == r.errorType() &&
           l.location() == r.location() &&
           l.sigmaPlus() == r.sigmaPlus() &&
           l.sigmaMinus() == r.sigmaMinus();
}

ase::AsymmetricEstimate operator*(const ase::AsymmetricEstimate& l,
                                  const double& r)
{
    if (r >= 0.0)
        return ase::AsymmetricEstimate(r*l.location(), r*l.sigmaPlus(),
                                       r*l.sigmaMinus(), l.errorType());
    else
        return ase::AsymmetricEstimate(r*l.location(), -r*l.sigmaMinus(),
                                       -r*l.sigmaPlus(), l.errorType());
}

ase::AsymmetricEstimate operator/(const ase::AsymmetricEstimate& l,
                                  const double& r)
{
    if (r == 0.0)
        throw std::invalid_argument(
            "In operator/ between ase::AsymmetricEstimate and double: "
            "division by zero encountered");
    return l*(1.0/r);
}
