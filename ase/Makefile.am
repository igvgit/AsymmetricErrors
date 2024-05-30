AM_CXXFLAGS = -O3

lib_LTLIBRARIES = libase.la

libase_la_SOURCES = AsymmetricEstimate.cc TruncatedDistribution1D.cc \
    DistributionModels1D.cc DiscretizedConvolution.cc AbsShiftableLogli.cc \
    GaussLegendreQuadrature.cc GeneralisedPoissonHelper.cc miscUtils.cc \
    specialFunctions.cc JohnsonCurves.cc mathUtils.cc EquidistantGrid.cc \
    SbMomentsBigGamma.cc SbMomentsCalculator.cc AbsLocationScaleFamily.cc \
    NumericalConvolution.cc Poly1D.cc CubicHermiteInterpolatorEG.cc \
    AbsLogLikelihoodCurve.cc fitSbParameters.cc GaussHermiteQuadrature.cc \
    LikelihoodCurveCopy.cc LikelihoodAccumulator.cc EdgeworthExpansion3.cc \
    LogLikelihoodCurves.cc InterpolatedDensity1D.cc HermiteProbOrthoPoly.cc \
    ZeroDerivsCubic.cc DoubleCubicInner.cc DoubleCubicLogspace.cc owens.cc \
    QuinticInner.cc QuinticLogspace.cc kullbackLeiblerDivergence.cc \
    densityIntegralGL.cc AbsDistributionModel1D.cc TabulatedDensity1D.cc \
    Gaussian.cc MixtureModel1D.cc statUtils.cc LegendreOrthoPoly1D.cc \
    LegendreDistro1D.cc PoissonLogli.cc

AM_CPPFLAGS = -I@top_srcdir@/

includedir = ${prefix}/include/ase

include_HEADERS = AbsDistributionModel1D.hh GaussLegendreQuadrature.hh \
    AbsShiftableLogli.hh AsymmetricEstimate.hh CPPRandomGen.hh mathUtils.hh \
    DistributionModels1D.hh specialFunctions.hh miscUtils.hh PoissonLogli.hh \
    DistributionFunctors1D.hh findRootUsingBisections.hh DoubleCubicInner.hh \
    AbsLocationScaleFamily.hh GaussHermiteQuadrature.hh DoubleFunctor1.hh \
    LikelihoodAccumulator.hh InterpolatedDensity1D.hh GaussianConvolution.hh \
    DistributionModel1DCopy.hh NumericalConvolution.hh SmoothDoubleCubic.hh \
    FunctorTimesShiftedX.hh AbsLogLikelihoodCurve.hh SbMomentsCalculator.hh \
    CubicHermiteInterpolatorEG.hh LogLikelihoodCurves.hh EquidistantGrid.hh \
    LikelihoodCurveCombination.hh GeneralisedPoissonHelper.hh statUtils.hh \
    PosteriorMomentFunctor.hh HermiteProbOrthoPoly.hh TransitionCubic.hh \
    ParabolicRailwayCurve.hh DerivativeFunctors.hh CRandomGen.hh Poly1D.hh \
    findMinimumGoldenSection.hh arrayStats.hh kullbackLeiblerDivergence.hh \
    DoubleCubicLogspace.hh QuinticInner.hh QuinticLogspace.hh Interval.hh \
    densityIntegralGL.hh gCdfValues.hh TabulatedDensity1D.hh AbsRNG.hh \
    DiscretizedConvolution.hh LikelihoodCurveCopy.hh ZeroDerivsCubic.hh \
    SymbetaDoubleIntegral.hh TDeriv.hh OPATGaussian.hh Gaussian.hh owens.hh \
    SymmetricBetaGaussian.hh MixtureModel1D.hh LegendreOrthoPoly1D.hh \
    LegendreDistro1D.hh TruncatedDistribution1D.hh
