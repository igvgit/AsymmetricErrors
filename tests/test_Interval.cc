#include "UnitTest++.h"

#include "test_utils.hh"

#include "ase/Interval.hh"

using namespace ase;

namespace {
    typedef Interval<int> Ii;

    void fill_palette(const Ii& interval, unsigned* palette,
                      const int maxlen)
    {
        int imin = interval.min();
        if (!interval.includesLeft())
            ++imin;
        int imax = interval.max() + 1;
        if (!interval.includesRight())
            --imax;
        for (int i=imin; i<imax; ++i)
            if (i >= 0 && i < maxlen)
                ++palette[i];
    }

    bool palettes_equal(const unsigned* palette1,
                        const unsigned* palette2,
                        const unsigned maxlen)
    {
        for (unsigned i=0; i<maxlen; ++i)
            if (palette1[i] != palette2[i])
                return false;
        return true;
    }

    Ii random_interval(const int maxlen)
    {
        int imin = maxlen*test_rng();
        if (imin >= maxlen)
            imin = maxlen - 1;
        int imax = maxlen*test_rng();
        if (imax >= maxlen)
            imax = maxlen - 1;
        const bool includeLeft = test_rng() < 0.5;
        const bool includeRight = test_rng() < 0.5;
        return Ii(imin, imax, Ii::fromInclusions(includeLeft, includeRight));
    }

    TEST(Interval_overlap)
    {
        const unsigned maxlen = 20;
        unsigned palette1[maxlen], palette2[maxlen], refPalette[maxlen];

        for (unsigned itry=0; itry<1000000; ++itry)
        {
            for (unsigned i=0; i<maxlen; ++i)
            {
                palette1[i] = 0;
                palette2[i] = 0;
                refPalette[i] = 0;
            }

            const Ii& in1 = random_interval(maxlen);
            CHECK_EQUAL(in1.type(), Ii::fromInclusions(in1.includesLeft(),
                                                       in1.includesRight()));
            const Ii& in2 = random_interval(maxlen);
            const Ii& inOver1 = in1.overlap(in2);
            const Ii& inOver2 = in2.overlap(in1);

            fill_palette(in1, refPalette, maxlen);
            fill_palette(in2, refPalette, maxlen);
            for (unsigned i=0; i<maxlen; ++i)
            {
                if (refPalette[i] == 2U)
                    refPalette[i] = 1U;
                else
                    refPalette[i] = 0U;
            }

            fill_palette(inOver1, palette1, maxlen);
            fill_palette(inOver2, palette2, maxlen);

            CHECK(palettes_equal(palette1, palette2, maxlen));
            CHECK(palettes_equal(refPalette, palette1, maxlen));
        }
    }
}
