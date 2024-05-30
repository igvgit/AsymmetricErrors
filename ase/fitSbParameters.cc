#include <cmath>
#include <cfloat>
#include <limits>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "ase/mathUtils.hh"
#include "ase/specialFunctions.hh"

#include "ase/DistributionModels1D.hh"
#include "ase/SbMomentsCalculator.hh"

#define SQRT2L 1.414213562373095048801689L
#define SQRPI  1.77245385090551602729816748L
#define N_B1_MAPPED   18U
#define MAPPED_DEG_LO 7U
#define MAPPED_DEG_HI 10U

// #define VERBOSE_ITERATIONS
// #define VERBOSE_ITERATION_COUNT

static long double inverseErf(const long double fval)
{
    typedef long double Real;
    Real x = ase::inverseGaussCdf((fval + 1.0L)/2.0L)/SQRT2L;
    for (unsigned i=0; i<2; ++i)
    {
        const Real guessed = erfl(x);
        const Real deri = 2.0L/SQRPI*expl(-x*x);
        x += (fval - guessed)/deri;
    }
    return x;
}

static long double minimumB(const long double skew)
{
    if (skew == 0.0L)
        return 0.0L;
    else
    {
        const long double x = skew*skew;
        return inverseErf(sqrtl(x/(4 + x)));
    }
}

static void lognormal_kurtosis_and_delta(const double B1,
                                         double *kurtosis,
                                         double *delta)
{
    if (B1 < 1.e-8)
    {
        const double limit_at_0 = 1.7777777777777777778;
        const double deriv_at_0 = 0.10699588477366255144;
        const double serv = B1*(1.0/9.0 - B1*(7.0/486.0 - B1*16.0/6561.0));
        *kurtosis = 3.0 + B1*(limit_at_0 + deriv_at_0*B1/2.0);
        *delta = sqrt(1.0/serv);
    }
    else
    {
        const double TMP=pow((2.0+B1+sqrt(B1*(4.0+B1)))/2.0, 1.0/3.0);
        const double W=TMP+1.0/TMP-1.0;
        *kurtosis = W*W*(W*(W+2.0)+3.0)-3.0;
        *delta = sqrt(1.0/log(W));
    }
}

static bool iterate_large_gamma(const ase::SbMomentsBigGamma& fitter,
                                const long double B1, const long double KURT,
                                long double *W, long double *Q)
{
    typedef long double Real;

    const unsigned maxiter = 1000;
    const Real tol = DBL_EPSILON;

    Real tryskew, trykurt, db1dw, db1dq, dkdw, dkdq;
    unsigned iter = 0;
    for (; iter < maxiter; ++iter)
    {
        if (*Q < 0.0L || *Q > 1.0L || *W < 1.0L)
            // Reached invalid parameter values.
            // This will not converge to anything reasonable.
            return false;

        if (!fitter.calculate(*W, *Q, 0, 0, &tryskew, &trykurt,
                              &db1dw, &db1dq, &dkdw, &dkdq))
            return false;
        db1dw *= (2.0L * tryskew);
        db1dq *= (2.0L * tryskew);
        tryskew *= tryskew;

        if (std::abs((trykurt - KURT)/KURT) < tol && 
            std::abs(tryskew - B1) < tol)
            // Looks like convergence
            break;

        if (db1dq > 0.0L || dkdq > 0.0L || db1dw < 0.0L || dkdw < 0.0L)
            // Invalid derivatives. Will not converge either.
            return false;

        const Real dtmp = db1dw*dkdq - db1dq*dkdw;
        if (dtmp == 0.0L)
            // Singular gradient. Bail out.
            return false;

        *Q -= (dkdw*(B1 - tryskew) + db1dw*(trykurt - KURT))/dtmp;
        *W -= (db1dq*(KURT - trykurt) + dkdq*(tryskew - B1))/dtmp;
    }
    return iter < maxiter;
}

static bool sbFitLargeGamma(const double skewness, const double kurtosis,
                            double *gamma, double *delta,
                            double *xlam, double *xi)
{
    typedef long double Real;

    const Real MAXRATIO = 1.5L;

    *gamma = 0.0;
    *delta = 0.0;
    *xlam = 0.0;
    *xi = 0.0;

    const Real KURT = kurtosis;
    const Real B1 = skewness*skewness;
    Real DTMP = powl((2.0L+B1+sqrtl(B1*(4.0L+B1)))/2.0L, 1.0L/3.0L);
    Real W = DTMP+1.0L/DTMP-1.0L;
    Real TRYKURT = W*W*(W*(W+2.0L)+3.0L)-3.0L;
    assert(kurtosis < TRYKURT && kurtosis > B1+1.0L);

    Real DKDQ0 = -4.0L*powl(W, 7.0L/2.0L)*(((((W+2.0L)*
           W+2.0L)*W+1.0L)*W-3.0L)*W-3.0L);
    Real DB1DQ0 = -6.0L*powl(W, 7.0L/2.0L)*(((W+2.0L)*W-1.0L)*W-2.0L);
    Real DB1DW0 = 3.0L*W*(2.0L + W);
    Real DKDW0 = W*(6.0L + 6.0L*W + 4.0L*W*W);
    Real Q = (KURT-TRYKURT)/DKDQ0;

    ase::SbMomentsBigGamma fitter;
    unsigned ideg = fitter.bestDeg(W, Q);
    if (ideg < 2)
        return false;
    if (ideg % 2)
        --ideg;
    fitter.setMaxDeg(ideg);

    // Check that at this value of Q the derivatives did not change
    // significantly (in this case we can hope that we are in
    // an approximately linear patch of space, and series in Q
    // should work). Note that the following test only works when
    // the highest degree included in the Q series is even.
    Real skew, kurt, db1dw, db1dq, dkdw, dkdq;
    if (!fitter.calculate(W, Q, 0, 0, &skew, &kurt,
                          &db1dw, &db1dq, &dkdw, &dkdq))
        return false;
    db1dw *= (2.0L * skew);
    db1dq *= (2.0L * skew);
    if (!(db1dq/DB1DQ0 > 1.0L/MAXRATIO && dkdq/DKDQ0 > 1.0L/MAXRATIO &&
          db1dw/DB1DW0 < MAXRATIO && dkdw/DKDW0 < MAXRATIO))
        return false;

    // Iterate to get the correct values of Q and W
    Q /= 2.0L;
    if (!iterate_large_gamma(fitter, B1, KURT, &W, &Q))
        return false;

    // Update the polynomial expansion degree and try to do this again
    unsigned newdeg = fitter.bestDeg(W, Q);
    if (newdeg < 2)
        return false;
    if (newdeg % 2)
        --newdeg;
    if (ideg != newdeg)
    {
        fitter.setMaxDeg(newdeg);
        if (!iterate_large_gamma(fitter, B1, KURT, &W, &Q))
            return false;
    }

    // Calculate the distribution parameters
    Real mean, var;
    fitter.calculate(W, Q, &mean, &var, &skew, &kurt,
                     &db1dw, &db1dq, &dkdw, &dkdq);
    if (var <= 0.0L)
        return false;

    Real lgamma, ldelta;
    fitter.getGammaDelta(W, Q, &lgamma, &ldelta);

    *gamma = lgamma;
    *delta = ldelta;
    *xlam = 1.0L/sqrtl(var);
    if (skewness >= 0.0)
        *xi = -*xlam*mean;
    else
    {
        *gamma = -*gamma;
        *xi = -*xlam*(1.0L - mean);
    }

    return true;
}

// The iteration starting point in the function below comes from Fortran code
// associated with ALGORITHM AS 99.2  APPL. STATIST. (1976) VOL.25, P.180.
static void initialGuessApplStat(const double skewness, const double kurtosis,
                                 long double* pG, long double* pD,
                                 long double* MAXDELTA)
{
    typedef long double Real;

    const Real ZERO  = 0.0L;
    const Real ONE   = 1.0L;
    const Real TWO   = 2.0L;
    const Real THREE = 3.0L;
    const Real HALF  = 0.5L;
    const Real QUART = 0.25L;

    const Real TT = 1.0e-10L;

    const Real A1 = 0.0124L;
    const Real A2 = 0.0623L;
    const Real A3 = 0.4043L;
    const Real A4 = 0.408L;
    const Real A5 = 0.479L;
    const Real A6 = 0.485L;
    const Real A7 = 0.5291L;
    const Real A8 = 0.5955L;
    const Real A9 = 0.626L;
    const Real A10 = 0.64L;
    const Real A11 = 0.7077L;
    const Real A12 = 0.7466L;
    const Real A13 = 0.8L;
    const Real A14 = 0.9281L;
    const Real A15 = 1.0614L;
    const Real A16 = 1.25L;
    const Real A17 = 1.7973L;
    const Real A18 = 1.8L;
    const Real A19 = 2.163L;
    const Real A20 = 2.5L;
    const Real A21 = 8.5245L;
    const Real A22 = 11.346L;

    const Real B2 = kurtosis;
    const Real RB1 = std::abs(skewness);
    const Real B1 = RB1 * RB1;

    // GET D AS FIRST ESTIMATE OF DELTA
    Real D = ZERO;
    Real E = B1 + ONE;
    Real X = HALF * B1 + ONE;
    Real Y = RB1 * sqrtl(QUART * B1 + ONE);
    Real U = powl(X + Y, ONE/THREE);
    Real W = U + ONE / U - ONE;
    Real F = W * W * (THREE + W * (TWO + W)) - THREE;

    // Make sure we are within the Sb region
    assert(B2 > E && B2 < F);

    *MAXDELTA = std::numeric_limits<double>::max();
    E = (B2 - E) / (F - E);
    if (RB1 > DBL_EPSILON)
        goto label5;
    F = TWO;
    goto label20;

label5:
    D = ONE/sqrtl(logl(W));
    *MAXDELTA = D;
    if (D < A10)
        goto label10;
    
    F = TWO - A21 / (D * (D * (D - A19) + A22));
    goto label20;

label10:
    F = A16 * D;

label20:
    F = E * F + ONE;
    if (F < A18)
        goto label25;
    D = (A9 * F - A4) * powl((THREE - F), (-A5));
    goto label30;

label25:
    D = A13 * (F - ONE);

label30:
    // GET G AS FIRST ESTIMATE OF GAMMA
    Real G = ZERO;
    if (B1 < TT)
        goto label70;
    if (D > ONE)
        goto label40;
    G = (A12 * powl(D, A17) + A8) * powl(B1, A6);
    goto label70;

label40:
    if (D <= A20)
        goto label50;
    U = A1;
    Y = A7;
    goto label60;

label50:
    U = A2;
    Y = A3;

label60:
    G = powl(B1, (U * D + Y)) * (A14 + D * (A15 * D - A11));

label70:
    *pG = G;
    *pD = D;
}

// static double highSkewDensityAt0(const double b1)
// {
//     static const double coeffs[] = {
//         -0.402399927378,
//         0.436556041241,
//         0.0547582097352,
//         -0.0121532352641,
//         0.000987160601653,
//         -2.97320984828e-05
//     };
//     static const double maxb1 = 3200.0;
//     static const double asymptoticSlope = 0.42;

//     const double x = log(b1 > maxb1 ? maxb1 : b1);
//     double logdens = polySeriesSum(
//         coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1U, x);
//     if (b1 > maxb1)
//         logdens += asymptoticSlope*log(b1/maxb1);
//     return exp(logdens);
// }

// static double highSkewDensityAt1(const double b1)
// {
//     static const double coeffs[] = {
//         1.22690212727,
//         -0.588575541973,
//         -0.313980579376,
//         0.11237283051,
//         -0.0199904069304,
//         0.00219813385047,
//         -0.0001558119111,
//         6.96057304594e-06,
//         -1.78762149972e-07,
//         2.01540828471e-09
//     };
//     static const double maxb1 = 1638400.0;
//     static const double asymptoticSlope = -0.42;

//     const double x = log(b1 > maxb1 ? maxb1 : b1);
//     double logdens = polySeriesSum(
//         coeffs, sizeof(coeffs)/sizeof(coeffs[0])-1U, x);
//     if (b1 > maxb1)
//         logdens += asymptoticSlope*log(b1/maxb1);
//     return exp(logdens);
// }

static double translated_map_log(
    const double x, const double bound,
    const double *coeff1, const unsigned deg1,
    const double *coeff2, const unsigned deg2)
{
    const double bound1 = 0.98*bound;
    const double bound2 = 1.02*bound;
    double value;
    if (x <= bound1)
        value = ase::polySeriesSum(coeff1, deg1, x);
    else if (x >= bound2)
        value = ase::polySeriesSum(coeff2, deg2, x);
    else
    {
        const double v1 = ase::polySeriesSum(coeff1, deg1, x);
        const double v2 = ase::polySeriesSum(coeff2, deg2, x);
        const double w = (x - bound1)/(bound2 - bound1);
        value = (1.0 - w)*v1 + w*v2;
    }
    return value;
}

static double highSkewFractionMap(const double b1, const double f0)
{
    static const double asymptoticSlope = 0.4;

    static const double mapped_b1[N_B1_MAPPED] = {
        12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0,
        6400.0, 12800.0, 25600.0, 51200.0, 102400.0, 204800.0, 409600.0,
        819200.0, 1638400.0
    };

    static const double breakpoints[N_B1_MAPPED] = {
        0.797500014305,
        0.532499998808,
        0.362500011921,
        0.262499988079,
        0.202500000596,
        0.167500004172,
        0.147500000894,
        0.132500000298,
        0.127499997616,
        0.132500000298,
        0.13750000298,
        0.152500003576,
        0.172499999404,
        0.192499995232,
        0.222499996424,
        0.252499997616,
        0.28749999404,
        0.317499995232
    };

    static const double coeffs1[N_B1_MAPPED][MAPPED_DEG_LO+1U] = {
        {
            0.88848721981,
            -1.59930121899,
            5.40149307251,
            -9.15358924866,
            8.85465526581,
            -3.40297651291,
            0.0,
            0.0
        },
        {
            1.25922334194,
            -1.6591053009,
            8.17307281494,
            -19.4381961823,
            26.7826061249,
            -15.1989564896,
            0.0,
            0.0
        },
        {
            1.61899662018,
            -1.75232994556,
            12.2115507126,
            -40.3946342468,
            78.7341842651,
            -64.6050567627,
            0.0,
            0.0
        },
        {
            1.96421635151,
            -1.89757275581,
            18.0296573639,
            -81.0305404663,
            216.780883789,
            -246.554718018,
            0.0,
            0.0
        },
        {
            2.29349112511,
            -2.11510944366,
            26.3449573517,
            -156.001953125,
            549.157531738,
            -821.070678711,
            0.0,
            0.0
        },
        {
            2.60707855225,
            -2.41698670387,
            37.6266899109,
            -279.009155273,
            1212.12426758,
            -2219.2434082,
            0.0,
            0.0
        },
        {
            2.90625166893,
            -2.81669902802,
            52.1897697449,
            -458.77746582,
            2307.61010742,
            -4838.75439453,
            0.0,
            0.0
        },
        {
            3.19278383255,
            -3.35033798218,
            71.5932769775,
            -729.010986328,
            4152.79003906,
            -9759.12792969,
            0.0,
            0.0
        },
        {
            3.46840548515,
            -4.00806474686,
            94.3102722168,
            -1049.57556152,
            6344.24755859,
            -15606.0498047,
            0.0,
            0.0
        },
        {
            3.73472547531,
            -4.77056312561,
            117.850631714,
            -1349.65246582,
            8104.75537109,
            -19462.3476562,
            0.0,
            0.0
        },
        {
            3.99331736565,
            -5.69363260269,
            145.283279419,
            -1702.88085938,
            10183.8828125,
            -23997.6933594,
            0.0,
            0.0
        },
        {
            4.24484777451,
            -6.59935951233,
            165.227325439,
            -1862.85449219,
            10426.7539062,
            -22652.546875,
            0.0,
            0.0
        },
        {
            4.48988103867,
            -7.42508602142,
            176.22769165,
            -1850.1237793,
            9443.25390625,
            -18492.3183594,
            0.0,
            0.0
        },
        {
            4.73748588562,
            -11.0991678238,
            372.054901123,
            -6712.28271484,
            71309.8984375,
            -435069.46875,
            1406202.0,
            -1862353.25
        },
        {
            4.97518014908,
            -12.6265249252,
            397.019836426,
            -6545.21582031,
            62161.71875,
            -334986.65625,
            949809.0625,
            -1098927.25
        },
        {
            5.20822620392,
            -14.0847883224,
            411.565612793,
            -6194.94140625,
            52964.765625,
            -254985.859375,
            643049.25,
            -659977.6875
        },
        {
            5.43607187271,
            -15.3069915771,
            408.866699219,
            -5541.44091797,
            42233.5664062,
            -180287.296875,
            401936.21875,
            -363997.40625
        },
        {
            5.65975284576,
            -16.6613864899,
            413.111358643,
            -5150.44238281,
            35879.125,
            -139529.921875,
            282851.3125,
            -232644.90625
        }
    };

    static const double coeffs2[N_B1_MAPPED][MAPPED_DEG_HI+1U] = {
        {
            -22.2461566925,
            133.51763916,
            -311.344573975,
            364.183074951,
            -213.05581665,
            50.0218315125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        },
        {
            -10.7667589188,
            117.579841614,
            -496.133422852,
            1159.28308105,
            -1614.46984863,
            1342.05322266,
            -616.983215332,
            121.147033691,
            0.0,
            0.0,
            0.0
        },
        {
            0.184769079089,
            15.9200782776,
            -80.6879348755,
            229.533920288,
            -381.608734131,
            373.717132568,
            -200.159240723,
            45.3910636902,
            0.0,
            0.0,
            0.0
        },
        {
            1.37426495552,
            8.11073970795,
            -54.4239196777,
            204.779251099,
            -398.103149414,
            246.051300049,
            591.685424805,
            -1565.12744141,
            1640.79150391,
            -853.795349121,
            181.477584839
        },
        {
            2.21858644485,
            -0.898729681969,
            17.5867576599,
            -121.824989319,
            560.753295898,
            -1680.24316406,
            3294.4440918,
            -4192.58544922,
            3336.79516602,
            -1509.07043457,
            296.127349854
        },
        {
            2.7269756794,
            -5.26850652695,
            63.0944786072,
            -382.437103271,
            1507.7911377,
            -3980.19726562,
            7084.10791016,
            -8379.53320312,
            6307.00292969,
            -2731.48999023,
            517.947814941
        },
        {
            3.13027143478,
            -8.16979789734,
            99.3108215332,
            -621.053283691,
            2477.47509766,
            -6553.11523438,
            11627.2363281,
            -13673.4921875,
            10216.8671875,
            -4389.31738281,
            825.285095215
        },
        {
            3.34741163254,
            -6.77058362961,
            89.4087982178,
            -584.873474121,
            2425.69995117,
            -6638.06689453,
            12134.2158203,
            -14647.2109375,
            11197.8925781,
            -4908.44677734,
            939.346435547
        },
        {
            3.62478923798,
            -7.14458656311,
            97.577331543,
            -656.225402832,
            2780.2109375,
            -7738.63330078,
            14344.1669922,
            -17516.2734375,
            13521.9853516,
            -5975.99023438,
            1151.60681152
        },
        {
            4.01652145386,
            -10.7784452438,
            141.13444519,
            -946.231506348,
            3981.71240234,
            -10991.0380859,
            20196.0449219,
            -24453.5625,
            18726.0722656,
            -8213.88769531,
            1571.77197266
        },
        {
            4.3929605484,
            -14.2432613373,
            181.899215698,
            -1216.21459961,
            5101.43701172,
            -14035.7197266,
            25709.8984375,
            -31040.328125,
            23708.1132812,
            -10374.5185547,
            1980.87133789
        },
        {
            5.3195400238,
            -31.4647808075,
            366.90335083,
            -2342.15576172,
            9410.51757812,
            -24897.53125,
            44025.2929688,
            -51499.5117188,
            38237.7578125,
            -16313.890625,
            3044.67602539
        },
        {
            5.89464092255,
            -39.6577262878,
            451.278686523,
            -2844.00244141,
            11318.6455078,
            -29738.2675781,
            52322.4335938,
            -60989.59375,
            45173.8710938,
            -19241.625,
            3587.2487793
        },
        {
            4.91162729263,
            -14.0745573044,
            216.528213501,
            -1624.49731445,
            7346.24707031,
            -21257.7851562,
            40312.609375,
            -49867.8359375,
            38751.8046875,
            -17168.5546875,
            3307.17700195
        },
        {
            4.39098501205,
            -6.59855318069,
            218.173873901,
            -1994.03759766,
            9764.94335938,
            -29225.9882812,
            56113.1914062,
            -69522.7578125,
            53804.0820312,
            -23668.6855469,
            4520.11523438
        },
        {
            -6.29440498352,
            187.634719849,
            -1287.72033691,
            4677.8515625,
            -8912.54003906,
            5212.37109375,
            13929.734375,
            -35869.4453125,
            37287.3632812,
            -19284.0507812,
            4072.21679688
        },
        {
            -39.5208435059,
            733.969665527,
            -5163.20410156,
            20331.7421875,
            -48572.6445312,
            70489.7109375,
            -55622.140625,
            9923.50292969,
            20982.0996094,
            -17427.8085938,
            4371.70556641
        },
        {
            -116.331077576,
            1880.78771973,
            -12485.5986328,
            46379.109375,
            -104308.390625,
            141145.875,
            -99461.25,
            4537.74853516,
            49886.0898438,
            -35893.25,
            8442.90625
        }
    };

    assert(b1 >= mapped_b1[0]);
    unsigned ib1_below = 0;
    for (unsigned i=0; i<N_B1_MAPPED; ++i)
    {
        if (b1 >= mapped_b1[i])
            ib1_below = i;
        else
            break;
    }

    const double mapbelow = translated_map_log(
        f0, breakpoints[ib1_below],
        &coeffs1[ib1_below][0], MAPPED_DEG_LO,
        &coeffs2[ib1_below][0], MAPPED_DEG_HI);
    double logmap;
    if (ib1_below == N_B1_MAPPED - 1U)
        logmap = mapbelow + asymptoticSlope*log(b1/mapped_b1[ib1_below]);
    else
    {
        const unsigned ib1_above = ib1_below + 1U;
        const double mapabove = translated_map_log(
            f0, breakpoints[ib1_above],
            &coeffs1[ib1_above][0], MAPPED_DEG_LO,
            &coeffs2[ib1_above][0], MAPPED_DEG_HI);
        const double w = 
            log(b1/mapped_b1[ib1_below])/
            log(mapped_b1[ib1_above]/mapped_b1[ib1_below]);
        logmap = w*mapabove + (1.0-w)*mapbelow;
    }
    const double mapvalue = exp(logmap);
    const double kappa = (1.0 - f0)/f0;
    return mapvalue/(mapvalue + kappa);
}

// Initial guess of delta for large values of skewness
static double highSkewDeltaGuess(const double skewness, const double kurtosis,
                                 double *deltamax)
{
    assert(deltamax);

    const double b1 = skewness*skewness;
    const double kmin = b1 + 1.0;
    double kmax;
    lognormal_kurtosis_and_delta(b1, &kmax, deltamax);
    const double f0 = (kurtosis - kmin)/(kmax - kmin);

    assert(f0 >= 0.0);
    assert(f0 <= 1.0);

    double frac = f0;
    if (f0 > 0.0 && f0 < 1.0)
        frac = highSkewFractionMap(b1, f0);
    return *deltamax * frac;
}

static double minPossibleSbGamma(const double b1)
{
    // We really need inverse erfc here...
    const double x = 0.5 * (1.0 - sqrt(b1/(4.0 + b1)));
    return ase::inverseGaussCdf(1.0 - x);
}

static double highSkewGammaGuess(const double skewness, const double delta)
{
    typedef long double Real;

    static const double tol = 1.0e-6;

    double gamma = minPossibleSbGamma(skewness*skewness);
    double previousGamma = gamma;

    Real a, b, mean, var, skew, kurt, db1db, db1da, dkurtdb, dkurtda;
    ase::SbMomentsMix fitter;
    fitter.getParameters(gamma, delta, &a, &b);
    bool status = fitter.calculate(
        a, b, &mean, &var, &skew, &kurt,
        &db1da, &db1db, &dkurtda, &dkurtdb);
    if (!status) throw std::runtime_error(
        "Error in highSkewGammaGuess (ase static function)");

    // Increase the gamma value by factor of 2 until we
    // exceed the given skewness
    while (skew < skewness)
    {
        previousGamma = gamma;
        gamma *= 2.0;
        fitter.getParameters(gamma, delta, &a, &b);
        status = fitter.calculate(
            a, b, &mean, &var, &skew, &kurt,
            &db1da, &db1db, &dkurtda, &dkurtdb);
        assert(status);
    }

    // Perform interval divisions to get a reasonable
    // gamma guess with the given tolerance
    while ((gamma - previousGamma)/previousGamma > tol)
    {
        const double trygamma = (gamma + previousGamma)/2.0;
        fitter.getParameters(trygamma, delta, &a, &b);
        status = fitter.calculate(
            a, b, &mean, &var, &skew, &kurt,
            &db1da, &db1db, &dkurtda, &dkurtdb);
        assert(status);
        if (skew < skewness)
            previousGamma = trygamma;
        else
            gamma = trygamma;
    }

    return gamma;
}

static bool sbFitGeneric(const double skewness, const double kurtosis,
                         double *gamma, double *delta,
                         double *xlam, double *xi)
{
    typedef long double Real;

    const double skewBoundary = 3.535534;
    const unsigned maxiter = 1000;
    const Real TOL = DBL_EPSILON;
    const Real SQTOL = sqrtl(TOL);

    const Real minB = minimumB(skewness);
    const Real B1 = skewness*skewness;
    const Real B2 = kurtosis;
    const bool useDoubleUpdate = B2 > 3.L - 2.L*B1;

    *gamma = 0.0;
    *delta = 0.0;
    *xlam = 0.0;
    *xi = 0.0;

    // Get the initial guess for delta and gamma
    Real G, D, MAXDELTA;
    if (skewness > skewBoundary)
    {
        double dmax;
        D = highSkewDeltaGuess(skewness, kurtosis, &dmax);
        G = highSkewGammaGuess(skewness, D);
        MAXDELTA = dmax;
    }
    else
        initialGuessApplStat(skewness, kurtosis, &G, &D, &MAXDELTA);

    unsigned nOscillations = 0;
    Real stepFactor = 1.L;
    Real DB1OLD = 0, DB2OLD = 0, DB1VOLD = 0, DB2VOLD = 0;
    Real a, b, mean, var, skew, kurt, db1db, db1da, dkurtdb, dkurtda;

    ase::SbMomentsMix fitter;
    fitter.getParameters(G, D, &a, &b);

    // Main iteration starts here
    unsigned M = 0;
    for (; M < maxiter; ++M)
    {
        const bool status = fitter.calculate(
            a, b, &mean, &var, &skew, &kurt,
            &db1da, &db1db, &dkurtda, &dkurtdb);
        db1db *= (2.0L * skew);
        db1da *= (2.0L * skew);

        if (!status)
        {
            std::cout << "WARNING in sbFitGeneric: "
                         "calculation of moments failed for skew "
                      << skewness << ", kurt " << kurtosis << std::endl;
            return false;
        }

        // Check for convergence
        const Real DB1 = skew*skew - B1;
        const Real DB2 = kurt - B2;

#ifdef VERBOSE_ITERATIONS
        std::cout << M << ": " << B1 << ' ' << B2 
                  << ' ' << a << ' ' << b
                  << ' ' << DB1 << ' ' << DB2
                  << ' ' << db1da << ' ' << db1db
                  << ' ' << dkurtda << ' ' << dkurtdb << std::endl;
#endif

        const Real absDB1 = std::abs(DB1);
        const Real absDB2 = std::abs(DB2);
        if (absDB1/(B1 + 1.L) < TOL && absDB2/(B2 + 1.L) < TOL)
            break;

        // Try to detect oscillations
        if (M > 3)
        {
            if (((DB1*DB1OLD < 0.L && absDB1 >= std::abs(DB1OLD)) || 
                 (DB1*DB1VOLD < 0.L && absDB1 >= std::abs(DB1VOLD))) && 
                ((DB2*DB2OLD < 0.L && absDB2 >= std::abs(DB2OLD)) || 
                 (DB2*DB2VOLD < 0.L && absDB2 >= std::abs(DB2VOLD))))
            {
                stepFactor /= SQRT2L;
                if (++nOscillations >= 10)
                {
                    std::cout << "WARNING in sbFitGeneric: "
                                 "oscillations detected for skew "
                              << skewness << ", kurt " << B2 << std::endl;
                    std::cout << "Current a is " << a << ", b is "
                              << b << std::endl;
                    std::cout << "Missing target skewness by " << std::abs(DB1)
                              << ", kurtosis by " << std::abs(DB2)
                              << std::endl;
                    return false;
                }
            }
        }
        DB1VOLD = DB1OLD;
        DB1OLD  = DB1;
        DB2VOLD = DB2OLD;
        DB2OLD  = DB2;

        // Update a and b
        Real U, Y;
        if (useDoubleUpdate)
        {
            if (M % 2)
            {
                // Kurtosis update cycle. Update kurtosis by changing
                // the value of b only. This is because kurtosis behavior
                // as a function of a is rather nasty: when b is above
                // 0.45 or so it becomes a multivalued function of a.
                if (db1da >= 0.L)
                {
                    // Abnormal case. Can only happen near the
                    // boundary  kurtosis == b1 + 1.
                    if (dkurtda > 0.0L)
                        U = DB2/dkurtda;
                    else
                        U = DB2/TOL;
                }
                else
                {
                    if (dkurtdb > 0.0L)
                        U = DB2/dkurtdb;
                    else
                        U = DB2/TOL;
                }
                Y = 0.0L;
            }
            else
            {
                // Skewness update cycle
                if (db1da >= 0.L)
                {
                    U = 0.0L;
                    assert(db1db);
                    Y = DB1/db1db;
                }
                else
                {
                    const Real det = db1db * dkurtda - db1da * dkurtdb;
                    if (std::abs(det/db1da) < SQTOL)
                    {
                        U = 0.0L;
                        Y = DB1/db1da;
                    }
                    else if (det)
                    {
                        U =  (dkurtda * DB1) / det;
                        Y = -(dkurtdb * DB1) / det;
                    }
                    else
                    {
                        std::cout << "WARNING in sbFitGeneric:"
                                  << " db1da is " << db1da
                                  << " db1db is " << db1db
                                  << " dkurtda is " << dkurtda
                                  << " dkurtdb is " << dkurtdb
                                  << " for skew " << skewness
                                  << ", kurt " << B2 << std::endl;
                        return false;
                    }
                }
            }
        }
        else
        {
            // We should be in a relatively linear patch of space
            const Real det = db1db * dkurtda - db1da * dkurtdb;
            if (det)
            {
                U = (dkurtda * DB1 - db1da * DB2) / det;
                Y = (db1db * DB2 - dkurtdb * DB1) / det;
            }
            else
            {
                if (DB1 == 0.0 && (dkurtda || dkurtdb))
                {
                    const Real sq = dkurtda*dkurtda + dkurtdb*dkurtdb;
                    U = DB2*dkurtdb/sq;
                    Y = DB2*dkurtda/sq;
                }
                else
                    assert(0);
            }
        }

        // Limit the changes in the parameters
        const Real UPRLIM = 2.L - (1.L/maxiter)*M;
        const Real LORLIM = powl(UPRLIM, 1.5L);
        if (B1)
        {
            const Real newb = b - stepFactor*U;
            if (b > 1.0L)
            {
                if (newb > b*UPRLIM*UPRLIM)
                    b *= (UPRLIM*UPRLIM);
                else if (newb < b/LORLIM)
                    b /= LORLIM;
                else
                    b = newb;
            }
            else if (newb > 2.0L)
                b = 2.0L;
            else
                b = newb;
            if (b < minB)
                b = minB;
        }
        else
            b = 0.L;

        const Real newa = a - stepFactor*Y;
        if (newa > a*UPRLIM)
            a *= UPRLIM;
        else if (newa < a/LORLIM)
            a /= LORLIM;
        else
            a = newa;
        if (a*SQRT2L > MAXDELTA)
            a = MAXDELTA/SQRT2L;
    }

    if (M >= maxiter)
    {
        std::cout << "WARNING in sbFitGeneric: "
                     "iterations failed to converge for skew "
                  << skewness << ", kurt " << B2 << std::endl;
        return false;
    }
    else
    {
#ifdef VERBOSE_ITERATION_COUNT
        std::cout << "******** sbFitGeneric: converged after " 
                  << M << " iterations" << std::endl;
#endif
        fitter.getGammaDelta(a, b, &G, &D);
        *delta = D;
        *xlam = 1.0L/sqrtl(var);
        if (skewness < 0.L)
        {
            *gamma = -G;
            *xi = -*xlam * (1.L - mean);
        }
        else
        {
            *gamma = G;
            *xi = -*xlam * mean;
        }
        return true;
    }
}

namespace ase {
    bool JohnsonSb::fitParameters(const double skew, const double kurt,
                                  double *gamma, double *delta,
                                  double *xlam, double *xi)
    {
        typedef long double Real;

        static const double high_a_coeffs[4] = {
            2.89695167542, 0.0207685492933,
            1.78114676476, 0.258008509874
        };
        const double max_high_a_skew = 0.730697;

        const Real tol = DBL_EPSILON;
        const unsigned maxiter = 1000U;

        assert(gamma);
        assert(delta);
        assert(xlam);
        assert(xi);

        *delta = 0.0;
        *xlam = 0.0;
        *gamma = 0.0;
        *xi = 0.0;        

        const double b1 = skew*skew;
        const double rb1 = fabs(skew);

        // Check that we are in the S_b range
        const double dtmp = pow(((2.0+b1+sqrt(b1*(4.0+b1)))/2.0), 1.0/3.0);
        const double w = dtmp + 1.0/dtmp - 1.0;
        const double lognormkurt = w*w*(w*(w+2.0)+3.0)-3.0;
        if (kurt >= lognormkurt || kurt <= b1 + 1.0)
            return false;

        // Process the special case of 0 skewness and high kurtosis.
        // Have to use 1-d iterations here instead of the standard 2-d.
        // The "SbMoments0SkewBigKurt" class expands the result in Taylor
        // series around 0 using x/delta as the small parameter.
        if (skew == 0.0 && kurt >= high_a_coeffs[0])
        {
            SbMoments0SkewBigKurt fitter;

            // Initial guess for the value of o = 1/a^2 = 2/delta^2
            const Real dtmp = 
                pow(55890.0 + sqrt(35478972.0+pow((55890.0-kurt*20736.0),2))
                    - kurt*20736.0, 1.0/3.0);
            Real asq = 0.375 - 10.866819/dtmp + dtmp/30.2381052;

            Real mean, var, tryskew, trykurt, dskewdo,
                dskewdp1, dkurtdo, dkurtdp1;

            unsigned niter = 0;
            for (; niter<maxiter; ++niter)
            {
                if (!fitter.calculate(
                    asq, 0.L, &mean, &var, &tryskew, &trykurt,
                    &dskewdo, &dskewdp1, &dkurtdo, &dkurtdp1))
                    throw std::runtime_error(
                        "Error 1 in ase::JohnsonSb::fitParameters");

                const Real diff = trykurt - kurt;
                if (std::abs(diff) <= tol)
                    break;
                asq -= diff/dkurtdo;
            }
            assert(niter < maxiter);

            Real lgamma, ldelta;
            fitter.getGammaDelta(asq, 0.L, &lgamma, &ldelta);

            *gamma = lgamma;
            *delta = ldelta;
            *xlam = 1.L/sqrtl(var);
            *xi = -*xlam * 0.5;

            return true;
        }

        // Check for small skewness and high kurtosis. We are still
        // going to use Taylor series around 0 using x/delta as the
        // small parameter. However, the series expansion is more
        // complicated than in the case of 0 skewness.
        if (rb1 <= max_high_a_skew)
        {
            const double bigakurt = polySeriesSum(
                high_a_coeffs,
                sizeof(high_a_coeffs)/sizeof(high_a_coeffs[0])-1, rb1);
            if (kurt >= bigakurt)
            {
                // Perform pure "large a" iterations
                SbMomentsBigDelta fitter;

                Real o, f0, trymean, tryvar, tryskew, trykurt;
                Real dskewdo, dskewdf0, dkurtdo, dkurtdf0;
                bool converged = false;

                {
                    // First approximation for o and f0
                    {
                        const double excess = kurt - 3.0;
                        double tsq;
                        if (excess != 0.0)
                        {
                            const double r = b1/excess;
                            tsq = 2.0*r/(2.0*r - 1.0)/9.0;
                        }
                        else
                            tsq = 1.0/9.0;
                        assert(tsq > 0.0);
                        if (tsq < 1.0)
                            f0 = (1 - sqrt(tsq))/2.0;
                        else
                        {
                            tsq = 1.0;
                            f0 = 5.0e-5;
                        }
                        o = 2.0*b1/9.0/tsq;
                    }

                    // Improve o and f0 by iterations
                    const double maxupratio = 2.0, maxdownratio = 3.0;
                    for (unsigned niter=0; niter<maxiter; ++niter)
                    {
                        if (!fitter.calculate(
                            o, f0, &trymean, &tryvar, &tryskew, &trykurt,
                            &dskewdo, &dskewdf0, &dkurtdo, &dkurtdf0))
                            throw std::runtime_error(
                                "Error 2 in ase::JohnsonSb::fitParameters");

                        const Real diffskew = tryskew - rb1;
                        const Real diffkurt = trykurt - kurt;
                        if (std::abs(diffskew) < tol && 
                            std::abs(diffkurt) < tol)
                        {
                            converged = true;
                            break;
                        }
                        const Real det = 
                            dkurtdo*dskewdf0 - dkurtdf0*dskewdo;
                        if (det == 0.0L)
                        {
                            // Null determinant. Can't iterate anymore.
                            // Hope that this can be solved by the generic
                            // routine which will use different variables.
                            break;
                        }

                        const Real new_o = 
                            o+(dkurtdf0*diffskew-dskewdf0*diffkurt)/det;
                        if (new_o > 0.25)
                            o = 0.25;
                        else if (new_o > o*maxupratio)
                            o *= maxupratio;
                        else if (new_o < o/maxdownratio)
                            o /= maxdownratio;
                        else
                            o = new_o;

                        const Real new_f0 = 
                            f0+(dskewdo*diffkurt-dkurtdo*diffskew)/det;
                        if (new_f0 > 0.5)
                            f0 = 0.5;
                        else if (new_f0 > f0*maxupratio)
                            f0 *= maxupratio;
                        else if (new_f0 < f0/maxdownratio)
                            f0 /= maxdownratio;
                        else
                            f0 = new_f0;
                    }
                }

                if (converged)
                {
                    assert(o > 0.0L);
                    *delta = 1.0L/sqrtl(o/2.0L);

                    assert(f0 >= 0.0L && f0 <= 0.5L);
                    if (f0 == 0.0L)
                        *gamma = *delta * 200.0;
                    else if (f0 == 0.5)
                        *gamma = 0.0;
                    else
                    {
                        *gamma = *delta * logl(1.0L/f0 - 1.0L);
                        if (*gamma < 0.0)
                            *gamma = 0.0;
                    }
                    assert(tryvar > 0.0L);
                    *xlam = 1.0L/sqrtl(tryvar);
                    if (skew > 0.0)
                        *xi = -*xlam * trymean;
                    else
                    {
                        *gamma *= -1.0;
                        *xi = -*xlam * (1.0L - trymean);
                    }
                    return true;
                }
                else
                {
                    std::cout << "WARNING in JohnsonSb::fitParameters: "
                                 "no \"large a\" convergence for skew = " 
                              << skew << ", kurt = " << kurt << std::endl;
                }
            }
        }

        // Try the "large gamma" mode
        if (sbFitLargeGamma(skew, kurt, gamma, delta, xlam, xi))
            return true;

        if (sbFitGeneric(skew, kurt, gamma, delta, xlam, xi))
            return true;

        // No fit available. This should not really happen.
        unsigned prec = std::cerr.precision(18);
        std::cerr << "ERROR in JohnsonSb::fitParameters: all algorithms "
            "failed for skew = " << skew << ", kurt = " << kurt << std::endl;
        std::cerr.precision(prec);

        return false;
    }
}
