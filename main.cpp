#include <iostream>
#include "numcxx.h"
using namespace std;

int main() {
  numcxx::NdArray<double> a({3, 3}, 0);
  cout << "Hello CMake." << endl;

  return 0;
}
