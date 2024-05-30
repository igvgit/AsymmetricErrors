#include <cmath>

#include "ase/HermiteProbOrthoPoly.hh"

static const long double sqrtl_table[101] = {
    0L, 1.L,
    1.414213562373095048801689L, 1.732050807568877293527446L, 2.L,
    2.236067977499789696409174L, 2.449489742783178098197284L, 
    2.645751311064590590501616L, 2.828427124746190097603377L, 3.L, 
    3.162277660168379331998894L, 3.316624790355399849114933L, 
    3.464101615137754587054893L, 3.605551275463989293119221L, 
    3.741657386773941385583749L, 3.872983346207416885179265L, 4.L, 
    4.12310562561766054982141L,  4.242640687119285146405066L, 
    4.358898943540673552236982L, 4.472135954999579392818347L, 
    4.582575694955840006588047L, 4.69041575982342955456563L, 
    4.795831523312719541597438L, 4.898979485566356196394568L, 5.L, 
    5.099019513592784830028224L, 5.196152422706631880582339L, 
    5.291502622129181181003232L, 5.38516480713450403125071L, 
    5.477225575051661134569698L, 5.567764362830021922119471L, 
    5.656854249492380195206755L, 5.744562646538028659850611L, 
    5.830951894845300470874153L, 5.916079783099616042567328L, 6.L, 
    6.082762530298219688999684L, 6.164414002968976450250192L, 
    6.244997998398398205846893L, 6.324555320336758663997787L, 
    6.403124237432848686488218L, 6.480740698407860230965967L, 
    6.55743852430200065234411L,  6.633249580710799698229865L, 
    6.708203932499369089227521L, 6.782329983125268139064556L, 
    6.855654600401044124935871L, 6.928203230275509174109785L, 7.L, 
    7.071067811865475244008444L, 7.1414284285428499979994L, 
    7.211102550927978586238443L, 7.280109889280518271097302L, 
    7.348469228349534294591852L, 7.416198487095662948711397L, 
    7.483314773547882771167497L, 7.549834435270749697236685L, 
    7.615773105863908285661411L, 7.681145747868608175769687L, 
    7.745966692414833770358531L, 7.810249675906654394129723L, 
    7.874007874011811019685034L, 7.937253933193771771504847L, 8.L, 
    8.062257748298549652366613L, 8.124038404635960360459884L, 
    8.185352771872449969953704L, 8.24621125123532109964282L, 
    8.306623862918074852584263L, 8.36660026534075547978172L, 
    8.42614977317635863063414L,  8.485281374238570292810132L, 
    8.544003745317531167871648L, 8.602325267042626771729474L, 
    8.660254037844386467637232L, 8.717797887081347104473964L, 
    8.774964387392122060406388L, 8.831760866327846854764043L, 
    8.888194417315588850091442L, 8.944271909999158785636695L, 9.L, 
    9.055385138137416626573808L, 9.110433579144298881945626L, 
    9.165151389911680013176094L, 9.219544457292887310002274L, 
    9.273618495495703752516416L, 9.327379053088815045554476L, 
    9.38083151964685910913126L,  9.43398113205660381132066L, 
    9.486832980505137995996681L, 9.539392014169456491526216L, 
    9.591663046625439083194876L, 9.643650760992954995760031L, 
    9.695359714832658028148881L, 9.746794344808963906838413L, 
    9.797958971132712392789136L, 9.848857801796104721746211L, 
    9.899494936611665341611821L, 9.949874371066199547344798L, 10.L
};

namespace ase {
    long double HermiteProbOrthoPoly::fast_sqrtl(const unsigned k)
    {
        if (k < sizeof(sqrtl_table)/sizeof(sqrtl_table[0]))
            return sqrtl_table[k];
        else
            return sqrtl(k);
    }

    long double HermiteProbOrthoPoly::weight(const long double x) const
    {
        const long double norm = 0.39894228040143267793994606L; // 1/sqrt(2 Pi)
        return norm*expl(-0.5*x*x);
    }

    long double HermiteProbOrthoPoly::weightIntegral(const long double x) const
    {
        return (1.0L + erfl(x/sqrtl_table[2]))/2.0L;
    }

    std::pair<long double,long double>
    HermiteProbOrthoPoly::recurrenceCoeffs(const unsigned k) const
    {
        long double b;
        if (k < sizeof(sqrtl_table)/sizeof(sqrtl_table[0]))
            b = sqrtl_table[k];
        else
            b = sqrtl(k);
        return std::pair<long double,long double>(0.0L, b);
    }

    long double HermiteProbOrthoPoly::poly(
        const unsigned degree, const long double x) const
    {
        long double polyk = 1.0L;
        if (degree)
        {
            long double polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
            }
        }
        return polyk;
    }

    void HermiteProbOrthoPoly::allpoly(const long double x,
                                       long double* values,
                                       const unsigned degree) const
    {
        assert(values);
        values[0] = 1.0L;
        if (degree)
        {
            long double polyk = 1.0L, polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
                values[k+1] = p;
            }
        }
    }

    long double HermiteProbOrthoPoly::series(
        const long double* coeffs, const unsigned degree,
        const long double x) const
    {
        assert(coeffs);
        long double sum = coeffs[0];
        if (degree)
        {
            long double polyk = 1.0L, polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                sum += p*coeffs[k+1];
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
            }
        }
        return sum;
    }

    long double HermiteProbOrthoPoly::weightedSeriesIntegral(
        const long double* coeffs,
        const unsigned degree, const long double x) const
    {
        assert(coeffs);
        long double sum = 0.0L;
        if (degree)
        {
            long double sqr, polyk = 1.0L, polykm1 = 0.0L;
            std::pair<long double,long double> rcurrent = recurrenceCoeffs(0);
            for (unsigned k=0; k<degree; ++k)
            {
                const std::pair<long double,long double>& rnext = recurrenceCoeffs(k+1);
                const long double p = ((x - rcurrent.first)*polyk -
                                       rcurrent.second*polykm1)/rnext.second;
                if (k+1U < sizeof(sqrtl_table)/sizeof(sqrtl_table[0]))
                    sqr = sqrtl_table[k + 1U];
                else
                    sqr = sqrtl(k + 1U);
                sum += polyk*coeffs[k+1]/sqr;
                polykm1 = polyk;
                polyk = p;
                rcurrent = rnext;
            }
        }
        long double sumTerm = 0.0L;
        if (sum)
            sumTerm = sum*weight(x);
        return coeffs[0]*weightIntegral(x) - sumTerm;
    }
}
