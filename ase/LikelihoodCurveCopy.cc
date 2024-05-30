#include "ase/LikelihoodCurveCopy.hh"
#include "ase/LikelihoodCurveCombination.hh"

using namespace ase;

LikelihoodCurveCopy operator+(
    const AbsLogLikelihoodCurve& l, const AbsLogLikelihoodCurve& r)
{
    return LikelihoodCurveCopy(new LikelihoodCurveSum(l, r));
}

LikelihoodCurveCopy operator-(
    const AbsLogLikelihoodCurve& l, const AbsLogLikelihoodCurve& r)
{
    return LikelihoodCurveCopy(new LikelihoodCurveDifference(l, r));
}
