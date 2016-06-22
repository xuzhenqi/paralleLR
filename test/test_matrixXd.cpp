#include "Eigen/Dense"
#include <iostream>
using namespace Eigen;
using namespace std;

class DummyReader {
 public:
  DummyReader() {
    temp_.resize(4, 2);
    temp_ << MatrixXd::Zero(2, 2),
          MatrixXd::Ones(2, 2);
    cout << "A() " << endl;
    cout << temp_ << endl;
  }
  void Dump() {
    cout << "Dump()" << endl;
    cout << temp_ << endl;
  }
 private:
  MatrixXd temp_;
};

int main() {
  A a;
  a.Dump();
  return 0;
}
