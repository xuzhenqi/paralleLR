#ifndef PARALLELR_MODEL_H
#define PARALLELR_MODEL_H
#include <random>
#include "Eigen/Dense"

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

namespace paralleLR {

class Model {
 public:
  virtual MatrixXd Classify(const MatrixXd& data) {}
  virtual double Loss(const MatrixXd& data, const MatrixXd& label) {}
  virtual void Gradient(const MatrixXd& data, const MatrixXd& label, 
                        MatrixXd& gw,
                        MatrixXd& gb) {}
  virtual void Update(const MatrixXd& deltaw, const MatrixXd& deltab) {}
  virtual double Accuracy(const MatrixXd& data, const MatrixXd& label) {}
  virtual const MatrixXd& w() {}
  virtual const MatrixXd& b() {}
  virtual ~Model() {}
};

class LRModel: public Model {
 public:
  LRModel(const MatrixXd& w, const MatrixXd& b): w_(w), b_(b) {}
  LRModel(int dim) {
    b_.resize(1, 1);
    b_(0, 0) = 0;
    w_.resize(dim, 1);
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id_);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes_nums_);
    if (node_id_ == 0) {
#endif
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d;
    for (int i = 0; i < dim; ++i) {
      w_(i, 0) = d(gen);
    }
#ifdef USE_MPI
    }
    MPI_Bcast(w_.data(), w_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }
  virtual MatrixXd Classify(const MatrixXd& data) {
    MatrixXd pred = Predict(data);
    MatrixXd zero = MatrixXd::Zero(pred.rows(), pred.cols());
    pred = (pred.array() < 0).select(zero, 1);
    return pred;
  }
  MatrixXd Predict(const MatrixXd& data) {
    MatrixXd pred = data * w_;
    for (int i = 0; i < pred.rows(); ++i) {
      pred(i, 0) += b_(0, 0);
    }
    return pred;
  }
  virtual double Loss(const MatrixXd& data, const MatrixXd& label) {
    MatrixXd pred = Predict(data);
    pred = Sigmoid(pred);
    MatrixXd loss = - label.array() * pred.array().log()
        - (1 - label.array()) * (1 - pred.array()).log();
    return loss.mean();
  }
  virtual void Gradient(const MatrixXd& data, const MatrixXd& label, 
                        MatrixXd& gw,
                        MatrixXd& gb) {
    MatrixXd pred = Predict(data);
    pred = Sigmoid(pred);
    gw = data.adjoint() * (pred - label) / label.rows();
    gb(0, 0) = (pred.array() - label.array()).mean();
  }
  virtual void Update(const MatrixXd& deltaw, const MatrixXd& deltab) {
    w_ -= deltaw;
    b_ -= deltab;
  }
  virtual double Accuracy(const MatrixXd& data, const MatrixXd& label) {
    MatrixXd pred = Classify(data);
    return  1 - (pred - label).squaredNorm() / label.rows();
  }
  const MatrixXd& w() { return w_; }
  const MatrixXd& b() { return b_; }
 private:
  MatrixXd Sigmoid(MatrixXd& x) {
    return ((-x).array().exp() + 1).cwiseInverse();
  }
  MatrixXd w_;
  MatrixXd b_;
#ifdef USE_MPI
  int node_id_;
  int nodes_nums_;
#endif
};

} // namespace paralleLR

#endif //PARALLELR_MODEL_H
