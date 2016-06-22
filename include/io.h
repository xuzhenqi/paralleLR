#ifndef PARALLELR_IO_H
#define PARALLELR_IO_H

#include <iostream>
#include <fstream>
#include <string>
#include "Eigen/Dense"
#ifdef USE_MPI
#include "mpi.h"
#endif
using namespace Eigen;
using namespace std;


namespace paralleLR {

class Reader {
 public:
  virtual void NextBatch(MatrixXd &data, MatrixXd &label) { }
  virtual ~Reader() { }
};

/*
 * DummyReader will produce Gaussian data with two class.
 */
class DummyReader: public Reader {
 public:
  DummyReader(const Vector2d &m0, const Matrix2d &s0,
              const Vector2d &m1, const Matrix2d &s1,
              int batch_size,
              int np = 1000, int nn = 1000) {
    Init(m0, s0, m1, s1, batch_size, np, nn);
  }
  // TODO:: The data range of Random()
  DummyReader(int batch_size, int np = 1000, int nn = 1000) {
    Vector2d m0(0, 0), m1(2, 2);
    Matrix2d s = Matrix2d::Identity();
    Init(m0, s, m1, s, batch_size, np, nn);
  }

  MatrixXd &Data() {
    return data_;
  }
  MatrixXd &Label() {
    return label_;
  }
  virtual void NextBatch(MatrixXd &data, MatrixXd &label);
  void Dump(string filename);

 private:

  void Init(const Vector2d &m0, const Matrix2d &s0,
       const Vector2d &m1, const Matrix2d &s1,
       int batch_size,
       int np, int nn);
  MatrixXd Gaussian2d(const Vector2d &mean, const Matrix2d &sigma, int num);
  void Shuffle(MatrixXd &data, MatrixXd &label);
  MatrixXd data_;
  MatrixXd label_;
  int size_;
  int next_item_;
  int batch_size_;
#ifdef USE_MPI
  int node_id_;
  int nodes_nums_;
#endif
};

}// namespace parallelLR

#endif //PARALLELR_IO_H
