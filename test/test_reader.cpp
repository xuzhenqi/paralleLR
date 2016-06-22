#include "matplotlibcpp.h"
#include "io.h"
#include <iostream>

namespace plt = matplotlibcpp;
using namespace paralleLR;
using namespace std;

int main() {
  DummyReader reader(100);
  reader.Dump("build/dataset.txt");
  MatrixXd& data = reader.Data();
  MatrixXd& label = reader.Label();
   // TODO: This code doesn't work under remote connection
  for(int i = 0; i < data.rows(); ++i) {
    if (label(i, 0) > 0.5) {
      plt::plot({data(i, 0)}, {data(i, 1)}, ".r");
    } else {
      plt::plot({data(i, 0)}, {data(i, 1)}, ".g");
    }
  }
  cout << "show" << endl;
  try {
    plt::show();
  } catch (runtime_error r) {
    cout << r.what() << endl;
  }
  return 0;
}