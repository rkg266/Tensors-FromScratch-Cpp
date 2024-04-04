#include <iostream>
#include <sstream>
#include <cstdlib>
#include <functional>
#include <string>
#include <map>
#include "Testing.hpp"

Device UseDevice = Device::cpu;

int main(int argc, const char* argv[]) {
    srand(time(NULL));
    std::string TestCommand = "MatrixMultiplication";
    Testing theTester{};

    if (TestCommand == "Indexing") {
        theTester.TestIndexing();
    }
    if (TestCommand == "ElementWiseOperations") {
        theTester.TestElementWiseOperations();
    }
    if (TestCommand == "MatrixMultiplication") {
        theTester.TestMatrixMultiplication();
    }

    return 0;
}