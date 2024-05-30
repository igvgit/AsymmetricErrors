#include <cmath>
#include <cfloat>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <limits>

#include "ase/Poly1D.hh"
#include "ase/mathUtils.hh"

static unsigned nSignChanges(const long double* seq, const unsigned len)
{
    assert(len);
    assert(seq);

    unsigned nChanges = 0;
    int sign = 100;
    for (unsigned i=0; i<len; ++i)
        if (seq[i])
        {
            const int newsign = seq[i] > 0.0L ? 1 : -1;
            if (sign != 100)
            {
                if (sign*newsign < 0)
                    ++nChanges;
            }
            sign = newsign;
        }
    return nChanges;
}

namespace ase {
    Poly1D::Poly1D(const std::vector<double>& c)
        : coeffs_(c.begin(), c.end())
    {
        if (coeffs_.empty())
            coeffs_.resize(1, 0.0L);
        else
            truncateLeadingZeros();
    }

    Poly1D::Poly1D(const std::vector<long double>& c)
        : coeffs_(c)
    {
        if (coeffs_.empty())
            coeffs_.resize(1, 0.0L);
        else
            truncateLeadingZeros();
    }

    Poly1D::Poly1D(const unsigned degree, const long double degCoeff)
        : coeffs_(degree+1U, 0.0L)
    {
        coeffs_[degree] = degCoeff;
        if (degree && degCoeff == 0.0L)
            coeffs_.resize(1);
    }

    Poly1D::Poly1D(const long double* coeffs, const unsigned degree)
        : coeffs_(coeffs, coeffs + (degree+1U))
    {
        assert(coeffs);
        truncateLeadingZeros();
    }

    Poly1D::Poly1D(const double* coeffs, const unsigned degree)
        : coeffs_(coeffs, coeffs + (degree+1U))
    {
        assert(coeffs);
        truncateLeadingZeros();
    }

    long double Poly1D::operator()(const long double x) const
    {
        return polySeriesSum(&coeffs_[0], coeffs_.size()-1U, x);
    }

    void Poly1D::valueAndDerivative(const long double x,
                                    long double* value,
                                    long double* derivative) const
    {
        polyAndDeriv(&coeffs_[0], coeffs_.size()-1U, x,
                     value, derivative);
    }

    void Poly1D::setCoefficient(const unsigned degree,
                                const long double value)
    {
        const unsigned myDeg = deg();
        if (degree <= myDeg)
        {
            coeffs_[degree] = value;
            if (degree == myDeg && !value)
                truncateLeadingZeros();
        }
        else if (value)
        {
            coeffs_.resize(degree + 1U);
            coeffs_[degree] = value;
        }
    }

    Poly1D Poly1D::derivative() const
    {
        const unsigned myDeg = deg();
        if (myDeg)
        {
            Poly1D result(myDeg-1U, myDeg*coeffs_[myDeg]);
            for (unsigned i=0; i<myDeg; ++i)
                result.coeffs_[i] = (i+1)*coeffs_[i+1];
            return result;
        }
        else
        {
            Poly1D dummy;
            return dummy;
        }
    }

    Poly1D Poly1D::integral(const long double c) const
    {
        const unsigned myDeg = deg();
        if (myDeg == 0U)
        {
            if (coeffs_[0])
            {
                long double cr[2];
                cr[0] = c;
                cr[1] = coeffs_[0];
                return Poly1D(cr, 1U);
            }
            else
                return c;
        }
        else
        {
            Poly1D result(myDeg+1U, coeffs_[myDeg]/(myDeg+1U));
            for (unsigned i=0; i<myDeg; ++i)
                result.coeffs_[i+1U] = coeffs_[i]/(i+1U);
            result.coeffs_[0] = c;
            return result;
        }
    }

    void Poly1D::truncate(const unsigned degree)
    {
        if (degree < deg())
        {
            coeffs_.resize(degree + 1U);
            truncateLeadingZeros();
        }
    }

    bool Poly1D::isClose(const Poly1D& r, const long double eps) const
    {
        if (eps < 0.0L) throw std::invalid_argument(
            "In ase::Poly1D::isClose: eps must not be negative");
        const unsigned maxDeg = std::max(deg(), r.deg());
        for (unsigned i=0; i<=maxDeg; ++i)
            if (std::abs(r[i] - (*this)[i]) > eps)
                return false;
        return true;
    }

    Poly1D Poly1D::operator-() const
    {
        Poly1D poly(*this);
        const unsigned len = coeffs_.size();
        for (unsigned i=0; i<len; ++i)
            poly.coeffs_[i] *= -1.0L;
        return poly;
    }

    void Poly1D::truncateLeadingZeros()
    {
        const unsigned sz = coeffs_.size();
        if (sz > 1U)
        {
            unsigned firstNon0 = sz;
            for (unsigned i=sz-1U; i>0; --i)
                if (coeffs_[i])
                {
                    firstNon0 = i;
                    break;
                }
            if (firstNon0 == sz)
                coeffs_.resize(1);
            else if (firstNon0 < sz-1U)
                coeffs_.resize(firstNon0+1U);
        }
    }

    Poly1D& Poly1D::operator*=(const Poly1D& r)
    {
        if (isNull())
            return *this;

        if (r.isNull())
        {
            coeffs_.resize(1);
            coeffs_[0] = 0.0L;
            return *this;
        }

        const unsigned mydeg = deg();
        const unsigned rdeg = r.deg();
        if (mydeg == 0U)
        {
            const long double c = coeffs_[0];
            coeffs_ = r.coeffs_;
            for (unsigned i=0; i<=rdeg; ++i)
                coeffs_[i] *= c;
        }
        else if (rdeg == 0U)
        {
            const long double c = r.coeffs_[0];
            for (unsigned i=0; i<=mydeg; ++i)
                coeffs_[i] *= c;
        }
        else
        {
            std::vector<long double> prod(mydeg + rdeg + 1U, 0.0L);
            for (unsigned i=0; i<=mydeg; ++i)
                for (unsigned j=0; j<=rdeg; ++j)
                    prod[i + j] += coeffs_[i]*r.coeffs_[j];
            coeffs_.swap(prod);
        }

        return *this;
    }

    Poly1D Poly1D::operator*(const Poly1D& r) const
    {
        if (isNull() || r.isNull())
        {
            Poly1D dummy;
            return dummy;
        }

        const unsigned mydeg = deg();
        const unsigned rdeg = r.deg();
        if (mydeg == 0U)
        {
            Poly1D p(r);
            const long double c = coeffs_[0];
            for (unsigned i=0; i<=rdeg; ++i)
                p.coeffs_[i] *= c;
            return p;
        }
        else if (rdeg == 0U)
        {
            Poly1D p(*this);
            const long double c = r.coeffs_[0];
            for (unsigned i=0; i<=mydeg; ++i)
                p.coeffs_[i] *= c;
            return p;
        }
        else
        {
            std::vector<long double> prod(mydeg + rdeg + 1U, 0.0L);
            for (unsigned i=0; i<=mydeg; ++i)
                for (unsigned j=0; j<=rdeg; ++j)
                    prod[i + j] += coeffs_[i]*r.coeffs_[j];
            return Poly1D(prod);
        }
    }

    Poly1D Poly1D::operator+(const Poly1D& r) const
    {
        const unsigned maxdeg = std::max(deg(), r.deg());
        Poly1D result(maxdeg, (*this)[maxdeg]+r[maxdeg]);
        for (unsigned i=0; i<maxdeg; ++i)
            result.coeffs_[i] = (*this)[i]+r[i];
        result.truncateLeadingZeros();
        return result;
    }

    Poly1D Poly1D::operator-(const Poly1D& r) const
    {
        const unsigned maxdeg = std::max(deg(), r.deg());
        Poly1D result(maxdeg, (*this)[maxdeg]-r[maxdeg]);
        for (unsigned i=0; i<maxdeg; ++i)
            result.coeffs_[i] = (*this)[i]-r[i];
        result.truncateLeadingZeros();
        return result;
    }

    Poly1D& Poly1D::operator+=(const Poly1D& r)
    {
        const unsigned maxdeg = std::max(deg(), r.deg());
        if (maxdeg > deg())
            coeffs_.resize(maxdeg + 1U);
        for (unsigned i=0; i<=maxdeg; ++i)
            coeffs_[i] += r[i];
        truncateLeadingZeros();
        return *this;
    }

    Poly1D& Poly1D::operator-=(const Poly1D& r)
    {
        const unsigned maxdeg = std::max(deg(), r.deg());
        if (maxdeg > deg())
            coeffs_.resize(maxdeg + 1U);
        for (unsigned i=0; i<=maxdeg; ++i)
            coeffs_[i] -= r[i];
        truncateLeadingZeros();
        return *this;
    }

    long double Poly1D::leadingCoefficient() const
    {
        const unsigned sz = coeffs_.size();
        assert(sz);
        const long double c = coeffs_.back();
        if (sz > 1U)
            assert(c);
        return c;
    }

    Poly1D Poly1D::operator/(const Poly1D& b) const
    {
        if (b.isNull()) throw std::invalid_argument(
            "In ase::Poly1D::operator/: division by zero encountered");

        const unsigned d = b.deg();
        const long double c = b.leadingCoefficient();
        if (d)
        {
            Poly1D q;
            Poly1D r(*this);
            if (r.deg() >= d)
                q.reserve(r.deg() - d);
            while (r.deg() >= d)
            {
                const Poly1D s(r.deg()-d, r.leadingCoefficient()/c);
                const Poly1D& sb = s*b;
                q += s;
                r -= sb;
                r.truncate(sb.deg() - 1U);
            }
            return q;
        }
        else
        {
            Poly1D q(*this);
            const unsigned mydeg = deg();
            for (unsigned i=0; i<=mydeg; ++i)
                q.coeffs_[i] /= c;
            return q;
        }
    }

    Poly1D Poly1D::operator%(const Poly1D& b) const
    {
        if (b.isNull()) throw std::invalid_argument(
            "In ase::Poly1D::operator%: division by zero encountered");

        const unsigned d = b.deg();
        if (d)
        {
            Poly1D r(*this);
            const long double c = b.leadingCoefficient();
            while (r.deg() >= d)
            {
                const Poly1D s(r.deg()-d, r.leadingCoefficient()/c);
                const Poly1D& sb = s*b;
                r -= sb;
                r.truncate(sb.deg() - 1U);
            }
            return r;
        }
        else
        {
            Poly1D dummy;
            return dummy;
        }
    }

    unsigned Poly1D::nRoots(const long double a, const long double b) const
    {
        if (a >= b) throw std::invalid_argument(
            "In ase::Poly1D::nRoots: invalid "
            "interval definition, must have a < b");
        const unsigned mydeg = deg();
        switch (mydeg)
        {
        case 0U:
            return 0U;

        case 1U:
        {
            const long double pa = (*this)(a);
            const long double pb = (*this)(b);
            const long double prod = pa*pb;
            if (prod < 0.0L)
                return 1U;
            else if (prod > 0.0L)
                return 0U;
            else if (pb == 0.0L)
                return 1U;
            else
                return 0U;
        }

        default:
        {
            // Construct the Sturm sequence
            std::vector<long double> buf(2U*(mydeg+1U));
            long double* avalues = &buf[0];
            long double* bvalues = avalues + (mydeg + 1U);

            Poly1D p_im1(*this);
            Poly1D p_i(derivative());
            Poly1D* ptr_im1 = &p_im1;
            Poly1D* ptr_i = &p_i;
            avalues[0] = p_im1(a);
            avalues[1] = p_i(a);
            bvalues[0] = p_im1(b);
            bvalues[1] = p_i(b);
            unsigned seqLength = 2U;
            for (;;)
            {
                *ptr_im1 = -(*ptr_im1 % *ptr_i);
                if (ptr_im1->isNull())
                    break;
                assert(seqLength <= mydeg);
                avalues[seqLength] = (*ptr_im1)(a);
                bvalues[seqLength++] = (*ptr_im1)(b);
                std::swap(ptr_im1, ptr_i);
            }
            const unsigned na = nSignChanges(avalues, seqLength);
            const unsigned nb = nSignChanges(bvalues, seqLength);
            assert(na >= nb);
            return na - nb;
        }
        }
    }

    long double Poly1D::singleRootOnInterval(
        long double a, long double fa, long double b, long double fb) const
    {
        static const long double tol = 4.0*std::numeric_limits<long double>::epsilon();
        static const long double sqrtol = sqrtl(tol);

        assert(fa*fb < 0.0);
        long double xmid = (a + b)/2.0L;
        const unsigned maxiter = 2000;
        for (unsigned iter=0; iter<maxiter; ++iter)
        {
            const long double fnew = (*this)(xmid);
            if (fnew == 0.0L)
                return xmid;
            if (fnew*fa > 0.0)
            {
                a = xmid;
                fa = fnew;
            }
            else
            {
                b = xmid;
                fb = fnew;
            }
            xmid = (a + b)/2.0L;
            if (std::abs(b - a)/(std::abs(xmid) + sqrtol) < tol)
            {
                long double f, fprime;
                const long double lolim = std::min(a, b);
                const long double uplim = std::max(a, b);
                for (unsigned iPolish=0; iPolish<2U; ++iPolish)
                {
                    this->valueAndDerivative(xmid, &f, &fprime);
                    if (fprime)
                    {
                        const long double xpolished = xmid - f/fprime;
                        if (lolim < xpolished && xpolished < uplim)
                            xmid = xpolished;
                    }
                }
                return xmid;
            }
        }
        throw std::runtime_error("In ase::Poly1D::singleRootOnInterval: "
                                 "iterations faled to converge");
        return 0.0L;
    }

    unsigned Poly1D::findRoots(const long double a, const long double b,
                               long double* roots) const
    {
        if (a >= b) throw std::invalid_argument(
            "In ase::Poly1D::findRoots: invalid "
            "interval definition, must have a < b");
        const unsigned mydeg = deg();
        switch (mydeg)
        {
        case 0U:
            return 0U;

        case 1U:
        {
            const long double rt = -coeffs_[0]/coeffs_[1];
            if (a < rt && rt < b)
            {
                roots[0] = rt;
                return 1U;
            }
            else
                return 0U;
        }

        case 2U:
        {
            long double rt[2];
            unsigned nRt2 = solveQuadratic(
                coeffs_[1]/coeffs_[2], coeffs_[0]/coeffs_[2],
                &rt[0], &rt[1]);
            if (nRt2 == 2U)
            {
                if (rt[0] == rt[1])
                    nRt2 = 1U;
                else if (rt[0] > rt[1])
                    std::swap(rt[0], rt[1]);
            }
            unsigned rtnum = 0;
            for (unsigned i=0; i<nRt2; ++i)
                if (a < rt[i] && rt[i] < b)
                    roots[rtnum++] = rt[i];
            return rtnum;
        }

        case 3U:
        {
            long double ldrt[3];
            unsigned irt3 = 1;
            {
                double v3[3];
                const unsigned nRt3 = solveCubic(
                    coeffs_[2]/coeffs_[3], coeffs_[1]/coeffs_[3],
                    coeffs_[0]/coeffs_[3], v3);
                assert(nRt3);
                if (nRt3 > 1U)
                    std::sort(v3, v3+nRt3);
                ldrt[0] = v3[0];
                for (unsigned i=1; i<nRt3; ++i)
                    if (v3[i] != v3[i-1U])
                        ldrt[irt3++] = v3[i];
            }
            // Polish the roots to reach long double precision
            const Poly1D& deriv = derivative();
            for (unsigned ir=0; ir<irt3; ++ir)
            {
                for (unsigned icycle=0; icycle<2; ++icycle)
                {
                    const long double der = deriv(ldrt[ir]);
                    if (der)
                    {
                        const long double f = (*this)(ldrt[ir]);
                        ldrt[ir] -= f/der;
                    }
                }
            }
            unsigned rtnum = 0;
            for (unsigned i=0; i<irt3; ++i)
                if (a < ldrt[i] && ldrt[i] < b)
                    roots[rtnum++] = ldrt[i];
            return rtnum;
        }

        default:
        {
            std::vector<long double> searchBuf(mydeg + 1U);
            long double* search = &searchBuf[0];
            search[0] = a;
            const unsigned nDerivRt = derivative().findRoots(a, b, search+1);
            const unsigned nIntervals = nDerivRt + 1U;
            search[nIntervals] = b;
            unsigned rtnum = 0;
            long double pleft = (*this)(a);
            for (unsigned iint=0; iint<nIntervals; ++iint)
            {
                const long double pright = (*this)(search[iint+1U]);
                if (pleft*pright > 0.0L)
                    continue;
                if (pleft*pright == 0.0L)
                {
                    if (pright == 0.0L && iint != nDerivRt)
                        roots[rtnum++] = search[iint+1U];
                }
                else
                {
                    roots[rtnum++] = singleRootOnInterval(
                        search[iint], pleft, search[iint+1U], pright);
                }
                pleft = pright;
            }
            return rtnum;
        }
        }
    }

    Poly1D Poly1D::monicDeg1(const long double b)
    {
        Poly1D mono(1, 1.0L);
        mono.coeffs_[0] = b;
        return mono;
    }

    Poly1D Poly1D::monicDeg2(const long double b, const long double c)
    {
        Poly1D mono(2, 1.0L);
        mono.coeffs_[0] = c;
        mono.coeffs_[1] = b;
        return mono;
    }
}
