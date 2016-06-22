#include <random>
#include <fstream>
#include <string>
#include "Eigen/Dense"
#include "io.h"

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;

namespace paralleLR {

void DummyReader::Init(const Vector2d &m0,
                       const Matrix2d &s0,
                       const Vector2d &m1,
                       const Matrix2d &s1,
                       int batch_size,
                       int np,
                       int nn) {
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &node_id_);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes_nums_);
  assert(batch_size % nodes_nums_ == 0);
  batch_size_ = batch_size / nodes_nums_;
  next_item_ = batch_size_ * node_id_;
  data_.resize(np + nn, 2);
  size_ = np + nn;
  label_.resize(np + nn, 1);
  if (node_id_ == 0) {
    data_ << Gaussian2d(m0, s0, np),
        Gaussian2d(m1, s1, nn);
    label_ << MatrixXd::Zero(np, 1),
        MatrixXd::Ones(nn, 1);
    Shuffle(data_, label_);
  }
  // TODO: don't know how Eigen::MatrixXd store the data, may have bugs.
  MPI_Bcast(data_.data(), data_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(label_.data(), label_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
  batch_size_ = batch_size;
  next_item_ = 0;
  size_ = np + nn;
  data_.resize(np + nn, 2);
  data_ << Gaussian2d(m0, s0, np),
      Gaussian2d(m1, s1, nn);
  label_.resize(np + nn, 1);
  label_ << MatrixXd::Zero(np, 1),
      MatrixXd::Ones(nn, 1);
  Shuffle(data_, label_);
#endif
}

void DummyReader::NextBatch(MatrixXd &data, MatrixXd &label) {
  data.resize(batch_size_, 2);
  label.resize(batch_size_, 1);
  for(int i = 0; i < batch_size_; ++i) {
    //cout << node_id_ << ": " << data_.rows() << " " << next_item_ << " " << size_ << " " << batch_size_ << " " << nodes_nums_ << endl;
    data(i, 0) = data_(next_item_, 0);
    data(i, 1) = data_(next_item_, 1);
    label(i, 0) = label_(next_item_, 0);
    ++next_item_;
    if (next_item_ == size_) {
      next_item_ = 0;
    }
  }
#ifdef USE_MPI
  // Thinking the data is an infinite searies.
  next_item_ += batch_size_ * (nodes_nums_ - 1);
  //cout << node_id_ << " Check0: " << next_item_ << endl;
  while (next_item_ >= size_)
    next_item_ -= size_;
  //cout << node_id_ << " Check1: " << next_item_ << endl;
#endif
}

MatrixXd DummyReader::Gaussian2d(const Vector2d &mean,
                             const Matrix2d &sigma,
                             int num) {
  MatrixXd data(num, 2);
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> d;
  for (int i = 0; i < num; ++i) {
    data(i, 0) = d(gen);
    data(i, 1) = d(gen);
  }
  double c = sqrt(sigma(1, 1));
  double b = sigma(0, 1) / c;
  double a = sqrt(sigma(0, 0) - b * b);
  Matrix2d trans;
  trans << a, 0,
          b, c;
  //cout << "data.size: " << data.rows() << " " << data.cols() << " mean.size: "
  //    << mean.rows() << " " << mean.cols() << endl;
  data = data * trans;
  data.rowwise() += mean.transpose();
  return data;
}

void DummyReader::Shuffle(MatrixXd &data, MatrixXd &label) {
  int rows = data.rows();
  random_device rd;
  mt19937 gen(rd());
  for (int i = rows - 1; i > 0; --i) {
    uniform_int_distribution<> d(0, i);
    int r = d(gen);
    swap(data(i, 0), data(r, 0));
    swap(data(i, 1), data(r, 1));
    swap(label(i, 0), label(r, 0));
  }
}

void DummyReader::Dump(string filename) {
  cout << data_.rows() << " " << data_.cols() << endl;
  ofstream of(filename);
  for (int i = 0; i < label_.rows(); ++i) {
    of << data_(i, 0) << " " << data_(i, 1) << " " << label_(i, 0) << endl;
  }
  of.close();
}

} // namespace paralleLR
