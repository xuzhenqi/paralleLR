#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "model.h"

using namespace Eigen;
using namespace paralleLR;
using namespace std;

MatrixXd data(4, 3), label(4, 1), w, b;
MatrixXd Predict(const MatrixXd& data) {
  MatrixXd out(data.rows(), 1);
  assert(data.cols() == w.rows());
  assert(b.size() == 1);
  for (int i = 0; i < data.rows(); ++i) {
    out(i, 0) = 0;
    for (int j = 0; j < data.cols(); ++j ) {
      out(i, 0) += data(i, j) * w(j, 0);
    }
    out(i, 0) += b(0, 0);
  }
  return out;
}

MatrixXd Classify(const MatrixXd& data) {
  MatrixXd pred = Predict(data);
  MatrixXd out(data.rows(), 1);
  assert(pred.cols() == 1);
  for (int i = 0; i < pred.rows(); ++i) {
    out(i, 0) = (pred(i, 0) < 0) ? 0 : 1;
  }
  return out;
}

double Loss(const MatrixXd& data, const MatrixXd& label) {
  MatrixXd pred = Predict(data);
  double loss = 0;
  for (int i = 0; i < pred.rows(); ++i) {
    pred(i, 0) = 1. / (exp(-pred(i, 0)) + 1); 
    if (label(i, 0) > 0.5) {
      loss -= log(pred(i, 0));
    } else {
      loss -= log(1 - pred(i, 0));
    }
  }
  return loss / data.rows();
}



bool Compare(const MatrixXd& m1, const MatrixXd& m2, double eps = 1e-5) {
  assert(m1.rows() == m2.rows());
  assert(m1.cols() == m2.cols());
  bool out = true;
  for (int i = 0; i < m1.rows(); ++i) {
    for (int j = 0; j < m1.cols(); ++j) {
      if (fabs(m1(i, j) - m2(i, j)) > eps) {
        cout << "Error: index(" << i << ", " << j << ") m1(i,j) = " << m1(i, j)
            << " m2(i,j) = " << m2(i, j) << endl;
        out = false;
      }
    }
  }
  return out;
}

bool Compare(double d1, double d2, double eps = 1e-5) {
  if (fabs(d1 - d2) > eps) {
    cout << "Error: d1 = " << d1 << " d2 = " << d2 << endl;
    return false;
  } else 
    return true;
}

int main() {
  data << 1, 2, 3, 
       4, 5, 6, 
       7, 8, 9, 
       10, 11, 12;
  label << 1,
        0,
        1,
        0;
  cout << "Test Construstor: " << endl;
  LRModel lr(3);
  cout << "w: " << lr.w() << endl;
  cout << "b: " << lr.b() << endl;
  w = lr.w();
  b = lr.b();
  cout << "Test Predict: " << endl;
  cout << Compare(Predict(data), lr.Predict(data)) << endl;
  cout << "Test Classify: " << endl;
  cout << Compare(Classify(data), lr.Classify(data)) << endl;
  cout << "Test Loss: " << endl;
  cout << Compare(Loss(data, label), lr.Loss(data, label)) << endl;
//  cout << "Test Gradient: " << endl;
//  cout << Compare(Gradient(data), lr.Gradient(data)) << endl;
//  cout << "Test Accuracy: " << endl;
//  cout << Compare(Accuracy(data, label), lr.Accuracy(data, label)) << endl;
  return 0;
}
