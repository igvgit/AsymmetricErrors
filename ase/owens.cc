#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "ase/owens.hh"

using namespace std;

//****************************************************************************80

inline static double gauss ( double t )

//****************************************************************************80
//
//  Purpose:
//
//    GAUSS returns the area of the lower tail of the normal curve.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 April 2012
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double T, the evaluation point.
//
//    Output, double GAUSS, the lower normal tail area.
//
{
  return ( 1.0 + erf ( t / M_SQRT2 ) ) / 2.0;
}

//****************************************************************************80

inline static double znorm2 ( double z )

//****************************************************************************80
//
//  Purpose:
//
//    ZNORM2 evaluates the normal CDF from Z to +oo.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    13 April 2012
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double Z, the lower limit.
//
//    Output, double ZNORM2, the probability that a standard
//    normal variable will lie between Z and +oo.
//
{
  return 0.5 * erfc ( z / M_SQRT2 );
}

//****************************************************************************80

inline static double znorm1 ( double z )

//****************************************************************************80
//
//  Purpose:
//
//    ZNORM1 evaluates the normal CDF from 0 to Z.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    13 April 2012
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double Z, the upper limit.
//
//    Output, double ZNORM1, the probability that a standard
//    normal variable will lie between 0 and Z.
//
{
  return 0.5 * erf ( z / M_SQRT2 );
}

namespace ase {
namespace Private {
//****************************************************************************80

void owen_values ( int &n_data, double &h, double &a, double &t )

//****************************************************************************80
//
//  Purpose:
//
//    OWEN_VALUES returns some values of Owen's T function.
//
//  Discussion:
//
//    Owen's T function is useful for computation of the bivariate normal
//    distribution and the distribution of a skewed normal distribution.
//
//    Although it was originally formulated in terms of the bivariate
//    normal function, the function can be defined more directly as
//
//      T(H,A) = 1 / ( 2 * pi ) *
//        Integral ( 0 <= X <= A ) e^(H^2*(1+X^2)/2) / (1+X^2) dX
//
//    In Mathematica, the function can be evaluated by:
//
//      fx = 1/(2*Pi) * Integrate [ E^(-h^2*(1+x^2)/2)/(1+x^2), {x,0,a} ]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 May 2009
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Mike Patefield, David Tandy,
//    Fast and Accurate Calculation of Owen's T Function,
//    Journal of Statistical Software,
//    Volume 5, Number 5, 2000, pages 1-25.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &H, a parameter.
//
//    Output, double &A, the upper limit of the integral.
//
//    Output, double &T, the value of the function.
//
{
# define N_MAX 28

  static double a_vec[N_MAX] = {
    0.2500000000000000E+00,
    0.4375000000000000E+00,
    0.9687500000000000E+00,
    0.0625000000000000E+00,
    0.5000000000000000E+00,
    0.9999975000000000E+00,
    0.5000000000000000E+00,
    0.1000000000000000E+01,
    0.2000000000000000E+01,
    0.3000000000000000E+01,
    0.5000000000000000E+00,
    0.1000000000000000E+01,
    0.2000000000000000E+01,
    0.3000000000000000E+01,
    0.5000000000000000E+00,
    0.1000000000000000E+01,
    0.2000000000000000E+01,
    0.3000000000000000E+01,
    0.5000000000000000E+00,
    0.1000000000000000E+01,
    0.2000000000000000E+01,
    0.3000000000000000E+01,
    0.5000000000000000E+00,
    0.1000000000000000E+01,
    0.2000000000000000E+01,
    0.3000000000000000E+01,
    0.1000000000000000E+02,
    0.1000000000000000E+03 };

  static double h_vec[N_MAX] = {
    0.0625000000000000E+00,
    6.5000000000000000E+00,
    7.0000000000000000E+00,
    4.7812500000000000E+00,
    2.0000000000000000E+00,
    1.0000000000000000E+00,
    0.1000000000000000E+01,
    0.1000000000000000E+01,
    0.1000000000000000E+01,
    0.1000000000000000E+01,
    0.5000000000000000E+00,
    0.5000000000000000E+00,
    0.5000000000000000E+00,
    0.5000000000000000E+00,
    0.2500000000000000E+00,
    0.2500000000000000E+00,
    0.2500000000000000E+00,
    0.2500000000000000E+00,
    0.1250000000000000E+00,
    0.1250000000000000E+00,
    0.1250000000000000E+00,
    0.1250000000000000E+00,
    0.7812500000000000E-02,
    0.7812500000000000E-02,
    0.7812500000000000E-02,
    0.7812500000000000E-02,
    0.7812500000000000E-02,
    0.7812500000000000E-02 };

  static double t_vec[N_MAX] = {
    3.8911930234701366E-02,
    2.0005773048508315E-11,
    6.3990627193898685E-13,
    1.0632974804687463E-07,
    8.6250779855215071E-03,
    6.6741808978228592E-02,
    0.4306469112078537E-01,
    0.6674188216570097E-01,
    0.7846818699308410E-01,
    0.7929950474887259E-01,
    0.6448860284750376E-01,
    0.1066710629614485E+00,
    0.1415806036539784E+00,
    0.1510840430760184E+00,
    0.7134663382271778E-01,
    0.1201285306350883E+00,
    0.1666128410939293E+00,
    0.1847501847929859E+00,
    0.7317273327500385E-01,
    0.1237630544953746E+00,
    0.1737438887583106E+00,
    0.1951190307092811E+00,
    0.7378938035365546E-01,
    0.1249951430754052E+00,
    0.1761984774738108E+00,
    0.1987772386442824E+00,
    0.2340886964802671E+00,
    0.2479460829231492E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    h = 0.0;
    a = 0.0;
    t = 0.0;
  }
  else
  {
    h = h_vec[n_data-1];
    a = a_vec[n_data-1];
    t = t_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double q ( double h, double ah )

//****************************************************************************80
//
//  Purpose:
//
//    Q computes (1/2) * p(H<Z) - T(H,AH).
//
//  Discussion:
//
//    The routine computes Q = (1/2) * P( H < Z ) - T ( H, AH ).
//
//    The result for Q is non-negative.
//
//    Warning : Q is computed as the difference between two terms;
//    When the two terms are of similar value this may produce
//    error in Q.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    04 February 2008
//
//  Author:
//
//    Original FORTRAN77 version by Mike Patefield, David Tandy.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Mike Patefield, David Tandy,
//    Fast and Accurate Calculation of Owen's T Function,
//    Journal of Statistical Software,
//    Volume 5, Number 5, 2000, pages 1-25.
//
//  Parameters:
//
//    Input, double H, the lower limit for Z.
//    0 < H.
//
//    Input, double AH, one of the arguments for the T function.
//
//    Output, double Q, the desired quantity.
//
{
  double ahh;
  double value;

  if ( 1.0 < ah )
  {
    ahh = ah * h;
    value = tfun ( ahh, 1.0 / ah, h ) - znorm2 ( ahh ) * znorm1 ( h );
  }
  else
  {
    value = 0.5 * znorm2 ( h ) - t ( h, ah );
  }

  return value;
}

//****************************************************************************80

double t ( double h, double a )

//****************************************************************************80
//
//  Purpose:
//
//    T computes Owen's T function for arbitrary H and A.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    13 April 2012
//
//  Author:
//
//    Original FORTRAN77 version by Mike Patefield, David Tandy.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Mike Patefield, David Tandy,
//    Fast and Accurate Calculation of Owen's T Function,
//    Journal of Statistical Software,
//    Volume 5, Number 5, 2000, pages 1-25.
//
//  Parameters:
//
//    Input, double H, A, the arguments.
//
//    Output, double T, the value of Owen's T function.
//
{
  double absa;
  double absh;
  double ah;
  double cut = 0.67;
  double normah;
  double normh;
  double value;

  absh = fabs ( h );
  absa = fabs ( a );
  ah = absa * absh;

  if ( absa <= 1.0 )
  {
    value = tfun ( absh, absa, ah );
  }
  else if ( absh <= cut )
  {
    value = 0.25 - znorm1 ( absh ) * znorm1 ( ah ) 
      - tfun ( ah, 1.0 / absa, absh );
  }
  else
  {
    normh = znorm2 ( absh );
    normah = znorm2 ( ah );
    value = 0.5 * ( normh + normah ) - normh * normah
    - tfun ( ah, 1.0 / absa, absh );
  }

  if ( a < 0.0 )
  {
    value = - value;
  }

  return value;
}
//****************************************************************************80

void t_test ( )

//****************************************************************************80
//
//  Purpose:
//
//    T_TEST demonstrates the use of T.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 December 2011
//
//  Author:
//
//    John Burkardt
//
{
  double a;
  double h;
  int n_data;
  double t1;
  double t2;

  cout << "\n";
  cout << "T_TEST:\n";
  cout << "  T computes Owen's T function.\n";
  cout << "  Compare to tabulated values.\n";
  cout << "\n";
  cout << "             H             A      "
       << "    T                         T  \n";
  cout << "                                  "
       << "    (Tabulated)               (TFN)               DIFF\n";
  cout << "\n";

  n_data = 0;

  for ( ; ; )
  {
    owen_values ( n_data, h, a, t1 );

    if ( n_data == 0 )
    {
      break;
    }

    t2 = t ( h, a );

    cout << "  " << setprecision(4) << setw(12) << h
         << "  " << setprecision(4) << setw(12) << a
         << "  " << setprecision(16) << setw(24) << t1
         << "  " << setprecision(16) << setw(24) << t2
         << "  " << setprecision(4) << setw(10) << fabs ( t1 - t2 ) << "\n";
  }

  return;
}
//****************************************************************************80

double tfun ( double h, double a, double ah )

//****************************************************************************80
//
//  Purpose:
//
//    TFUN computes Owen's T function for a restricted range of parameters.
//
//  Discussion:
//
//    This routine computes Owen's T-function of H and A.
//
//    This routine, originally named "TF", was renamed "TFUN" to avoid
//    a conflict with a built in MATLAB function.
//
//    Thanks to Marko Javenpaa for identifying an incorrect modification of
//    the code that resulting in halving the output of ZNORM1 for algorithms
//    T2 and T3.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    13 July 2017
//
//  Author:
//
//    Original FORTRAN77 version by Mike Patefield, David Tandy.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Mike Patefield, David Tandy,
//    Fast and Accurate Calculation of Owen's T Function,
//    Journal of Statistical Software,
//    Volume 5, Number 5, 2000, pages 1-25.
//
//  Parameters:
//
//    Input, double H, the H argument of the function.
//    0 <= H.
//
//    Input, double A, the A argument of the function.
//    0 <= A <= 1.
//
//    Input, double AH, the value of A*H.
//
//    Output, double TF, the value of Owen's T function.
//
{
  double ai;
  double aj;
  double arange[7] = {
    0.025, 0.09, 0.15, 0.36, 0.5,
    0.9, 0.99999 };
  double as;
  double c2[21] = {
                                   0.99999999999999987510,
     -0.99999999999988796462,      0.99999999998290743652,
     -0.99999999896282500134,      0.99999996660459362918,
     -0.99999933986272476760,      0.99999125611136965852,
     -0.99991777624463387686,      0.99942835555870132569,
     -0.99697311720723000295,      0.98751448037275303682,
     -0.95915857980572882813,      0.89246305511006708555,
     -0.76893425990463999675,      0.58893528468484693250,
     -0.38380345160440256652,      0.20317601701045299653,
     -0.82813631607004984866E-01,  0.24167984735759576523E-01,
     -0.44676566663971825242E-02,  0.39141169402373836468E-03 };
  double dhs;
  double dj;
  double gj;
  double hrange[14] = {
    0.02, 0.06, 0.09, 0.125, 0.26,
    0.4,  0.6,  1.6,  1.7,   2.33,
    2.4,  3.36, 3.4,  4.8 };
  double hs;
  int i;
  int iaint;
  int icode;
  int ihint;
  int ii;
  int j;
  int jj;
  int m;
  int maxii;
  int meth[18] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6 };
  double normh;
  int ord[18] = {
    2, 3, 4, 5, 7,10,12,18,10,20,30,20, 4, 7, 8,20,13, 0 };
  double pts[13] = {
                                   0.35082039676451715489E-02,
      0.31279042338030753740E-01,  0.85266826283219451090E-01,
      0.16245071730812277011,      0.25851196049125434828,
      0.36807553840697533536,      0.48501092905604697475,
      0.60277514152618576821,      0.71477884217753226516,
      0.81475510988760098605,      0.89711029755948965867,
      0.95723808085944261843,      0.99178832974629703586 };
  double r;
  double rrtpi = 0.39894228040143267794;
  double rtwopi = 0.15915494309189533577;
  int select[15*8] = {
    1, 1, 2,13,13,13,13,13,13,13,13,16,16,16, 9,
    1, 2, 2, 3, 3, 5, 5,14,14,15,15,16,16,16, 9,
    2, 2, 3, 3, 3, 5, 5,15,15,15,15,16,16,16,10,
    2, 2, 3, 5, 5, 5, 5, 7, 7,16,16,16,16,16,10,
    2, 3, 3, 5, 5, 6, 6, 8, 8,17,17,17,12,12,11,
    2, 3, 5, 5, 5, 6, 6, 8, 8,17,17,17,12,12,12,
    2, 3, 4, 4, 6, 6, 8, 8,17,17,17,17,17,12,12,
    2, 3, 4, 4, 6, 6,18,18,18,18,17,17,17,12,12 };
  double value = 0.0;
  double vi;
  double wts[13] = {
                                   0.18831438115323502887E-01,
      0.18567086243977649478E-01,  0.18042093461223385584E-01,
      0.17263829606398753364E-01,  0.16243219975989856730E-01,
      0.14994592034116704829E-01,  0.13535474469662088392E-01,
      0.11886351605820165233E-01,  0.10070377242777431897E-01,
      0.81130545742299586629E-02,  0.60419009528470238773E-02,
      0.38862217010742057883E-02,  0.16793031084546090448E-02 };
  double y;
  double yi;
  double z;
  double zi;
//
//  Determine appropriate method from t1...t6
//
  ihint = 15;

  for ( i = 1; i <= 14; i++ )
  {
    if ( h <= hrange[i-1] )
    {
      ihint = i;
      break;
    }
  }

  iaint = 8;

  for ( i = 1; i <= 7; i++ )
  {
    if ( a <= arange[i-1] )
    {
      iaint = i;
      break;
    }
  }

  icode = select[ihint-1+(iaint-1)*15];
  m = ord[icode-1];
//
//  t1(h, a, m) ; m = 2, 3, 4, 5, 7, 10, 12 or 18
//  jj = 2j - 1 ; gj = exp(-h*h/2) * (-h*h/2)**j / j//
//  aj = a**(2j-1) / (2*pi)
//
  if ( meth[icode-1] == 1 )
  {
    hs = - 0.5 * h * h;
    dhs = exp ( hs );
    as = a * a;
    j = 1;
    jj = 1;
    aj = rtwopi * a;
    value = rtwopi * atan ( a );
    dj = dhs - 1.0;
    gj = hs * dhs;

    for ( ; ; )
    {
      value = value + dj * aj / ( double ) ( jj );

      if ( m <= j )
      {
        return value;
      }
      j = j + 1;
      jj = jj + 2;
      aj = aj * as;
      dj = gj - dj;
      gj = gj * hs / ( double ) ( j );
    }
  }
//
//  t2(h, a, m) ; m = 10, 20 or 30
//  z = (-1)**(i-1) * zi ; ii = 2i - 1
//  vi = (-1)**(i-1) * a**(2i-1) * exp[-(a*h)**2/2] / sqrt(2*pi)
//
  else if ( meth[icode-1] == 2 )
  {
    maxii = m + m + 1;
    ii = 1;
    value = 0.0;
    hs = h * h;
    as = - a * a;
    vi = rrtpi * a * exp ( - 0.5 * ah * ah );
    z = znorm1 ( ah ) / h;
    y = 1.0 / hs;

    for ( ; ; )
    {
      value = value + z;

      if ( maxii <= ii )
      {
        value = value * rrtpi * exp ( - 0.5 * hs );
        return value;
      }
      z = y * ( vi - ( double ) ( ii ) * z );
      vi = as * vi;
      ii = ii + 2;
    }
  }
//
//  t3(h, a, m) ; m = 20
//  ii = 2i - 1
//  vi = a**(2i-1) * exp[-(a*h)**2/2] / sqrt(2*pi)
//
  else if ( meth[icode-1] == 3 )
  {
    i = 1;
    ii = 1;
    value = 0.0;
    hs = h * h;
    as = a * a;
    vi = rrtpi * a * exp ( - 0.5 * ah * ah );
    zi = znorm1 ( ah ) / h;
    y = 1.0 / hs;

    for ( ; ; )
    {
      value = value + zi * c2[i-1];

      if ( m < i )
      {
        value = value * rrtpi * exp ( - 0.5 * hs );
        return value;
      }
      zi = y  * ( ( double ) ( ii ) * zi - vi );
      vi = as * vi;
      i = i + 1;
      ii = ii + 2;
    }
  }
//
//  t4(h, a, m) ; m = 4, 7, 8 or 20;  ii = 2i + 1
//  ai = a * exp[-h*h*(1+a*a)/2] * (-a*a)**i / (2*pi)
//
  else if ( meth[icode-1] == 4 )
  {
    maxii = m + m + 1;
    ii = 1;
    hs = h * h;
    as = - a * a;
    value = 0.0;
    ai = rtwopi * a * exp ( - 0.5 * hs * ( 1.0 - as ) );
    yi = 1.0;

    for ( ; ; )
    {
      value = value + ai * yi;

      if ( maxii <= ii )
      {
        return value;
      }
      ii = ii + 2;
      yi = ( 1.0 - hs * yi ) / ( double ) ( ii );
      ai = ai * as;
    }
  }
//
//  t5(h, a, m) ; m = 13
//  2m - point gaussian quadrature
//
  else if ( meth[icode-1] == 5 )
  {
    value = 0.0;
    as = a * a;
    hs = - 0.5 * h * h;
    for ( i = 1; i <= m; i++ )
    {
      r = 1.0 + as * pts[i-1];
      value = value + wts[i-1] * exp ( hs * r ) / r;
    }
    value = a * value;
  }
//
//  t6(h, a);  approximation for a near 1, (a<=1)
//
  else if ( meth[icode-1] == 6 )
  {
    normh = znorm2 ( h );
    value = 0.5 * normh * ( 1.0 - normh );
    y = 1.0 - a;
    r = atan ( y / ( 1.0 + a ) );

    if ( r != 0.0 )
    {
      value = value - rtwopi * r * exp ( - 0.5 * y * h * h / r );
    }
  }
  return value;
}
} // namespace Private
} // namespace ase
