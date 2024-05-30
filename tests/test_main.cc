#include <iostream>

#include "UnitTest++.h"
#include "TestReporterStdout.h"
#include "CmdLine.hh"

int main(int argc, char const* argv[])
{
    CmdLine cmdline(argc, argv);
    std::cout << "Running " << cmdline.progname()
              << " test suite." << std::endl;
    return UnitTest::RunAllTests();
}
