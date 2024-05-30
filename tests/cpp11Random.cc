#include <iostream>
#include <random>

#include "test_utils.hh"
#include "CmdLine.hh"

#include "ase/CPPRandomGen.hh"

using namespace ase;
using namespace std;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " npoints\n" << endl;
}

int main(int argc, char const* argv[])
{
    CmdLine cmdline(argc, argv);
    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    // Parse command line arguments
    unsigned npoints;

    try {
        cmdline.optend();
        if (cmdline.argc() != 1U)
            throw CmdLineError("wrong number of command line arguments");

        cmdline >> npoints;
    }
    catch (const CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": "
             << e.str() << endl;
        return 1;
    }

    random_device rd;
    mt19937_64 eng(rd());
    CPPRandomGen<decltype(eng)> gen(eng);

    for (unsigned i=0; i<npoints; ++i)
        cout << gen() << endl;

    return 0;
}
