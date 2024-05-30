# This test suite is utilizing UnitTest++, so it must be installed
# on your computer before compiling and running the tests for the
# AsymmetricErrors package.
#
# Run "make" in order to run the tests.
# Run "make binaries" in order to compile example programs.
#
# Set the following variable correctly:
UTESTPP_DIR = /home/igv/Code/UnitTest++

OFILES_COMMON = test_utils.o

OTESTS_ALL = test_Poly1D.o test_DistributionModels1D.o test_Convolution.o \
              test_CubicHermiteInterpolatorEG.o test_LogLikelihoodCurves.o \
              test_HermiteProbOrthoPoly.o test_TransitionCubic.o \
              test_ParabolicRailwayCurve.o test_Interval.o test_mathUtils.o \
              test_findMinimumGoldenSection.o test_ZeroDerivsCubic.o \
              test_kullbackLeiblerDivergence.o test_statUtils.o

# OPTIMIZE = -std=c++11 -g -ggdb -O0
OPTIMIZE = -std=c++11 -O3

INCLUDES = -I. -I.. -I$(UTESTPP_DIR)/src
CPPFLAGS = $(OPTIMIZE) $(INCLUDES) -Wall -W -Werror -Wno-unused-parameter

LIBS = -L../ase/.libs -lase -L$(UTESTPP_DIR) -lUnitTest++ -lm

%.o : %.cc
	g++ -c $(CPPFLAGS) -fPIC -MD $< -o $@
	@sed -i 's,\($*\.o\)[:]*\(.*\),$@: $$\(wildcard\2\)\n\1:\2,g' $*.d

all: main run_all

main: test_main.o $(OTESTS_ALL) $(OFILES_COMMON)
	rm -f $@
	g++ $(OPTIMIZE) -fPIC -o $@ $^ $(LIBS)

run_all: main
	./main

PROGRAMS = cpp11Random.cc

BINARIES = $(PROGRAMS:.cc=)

binaries: $(BINARIES)

$(BINARIES): % : %.o $(OFILES_COMMON); g++ $(OPTIMIZE) -fPIC -o $@ $^ $(LIBS)

clean:
	rm -f tests $(BINARIES) *.out core.* *.o *.d *~

-include test_main.d
-include $(OFILES_COMMON:.o=.d)
-include $(OTESTS_ALL:.o=.d)
-include $(PROGRAMS:.cc=.d)
