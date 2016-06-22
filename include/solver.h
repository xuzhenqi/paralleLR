#ifndef PARALLELR_SOLVER_H
#define PARALLELR_SOLVER_H
#include <memory>
#include <iostream>
#include "model.h"
#include "io.h"

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;

namespace paralleLR {

class Solver {
 public:
  virtual void Solve() {}
  virtual ~Solver() {}
};

struct SolverParameter {
  double epsilon_;
  int max_iter_;
  double lr_;
  SolverParameter(): epsilon_(1e-6), max_iter_(1000),
                     lr_(0.01) {}
};

class SGDSolver : public Solver {
 public:
  SGDSolver(SolverParameter& param, shared_ptr<Reader> reader,
            shared_ptr<Model> model) {
    epsilon_ = param.epsilon_;
    max_iter_ = param.max_iter_;
    lr_ = param.lr_;
    model_ = model;
    reader_ = reader;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id_);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes_nums_);
#endif
  }

  virtual void Solve() {
#ifdef USE_MPI
    // TODO: change MPIc to MPIc++
    MatrixXd other_gw, other_gb;
    MPI_Status status;
#endif
    MatrixXd data, label, gw;
    MatrixXd gb(1, 1);
    double pre_loss, cur_loss;
    for (int iter = 0; iter < max_iter_; ++iter) {
      reader_->NextBatch(data, label);
      //cout << node_id_ << " NextBatch: " << endl;
      model_->Gradient(data, label, gw, gb);
      //cout << node_id_ <<  " Gradient: " << endl;
#ifdef USE_MPI
      if (node_id_ == 0) {
        other_gw.resize((nodes_nums_ - 1) * gw.rows(), 1);
        other_gb.resize((nodes_nums_ - 1) * gb.rows(), 1);
        for (int i = 1; i < nodes_nums_; ++i) {
          MPI_Recv(other_gw.data() + gw.rows() * (i - 1), gw.rows(), 
                   MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
          MPI_Recv(other_gb.data() + gb.rows() * (i - 1), gb.rows(),
                   MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        }
      } else {
        MPI_Send(gw.data(), gw.rows(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(gb.data(), gb.rows(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
      if (node_id_ == 0) {
        for (int i = 1; i < nodes_nums_; ++i) {
          for (int j = 0; j < gw.rows(); ++j) {
            gw(j, 0) += other_gw((i - 1) * gw.rows() + j, 0);
          }
          gb(0, 0) += other_gb((i - 1) * gb.rows(), 0);
        }
        gw /= nodes_nums_;
        gb /= nodes_nums_;
      }
      MPI_Bcast(gw.data(), gw.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(gb.data(), gb.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      MatrixXd deltaw = gw * lr_, deltab = gb * lr_;
      model_->Update(deltaw, deltab);
      // TODO: using DLOG and LOG instead of cout 
      cout << node_id_ << " Iteration " << iter << " loss: " << model_->Loss(data, label)
          << endl;
      //cout << node_id_ << " deltaw " << deltaw(0, 0) << " " << deltaw(1, 0) << " deltab: " << deltab << endl;
      if (IsStable(deltaw, deltab))
        break;
    }
  }

 private:
  bool IsStable(MatrixXd & deltaw, MatrixXd & deltab) {
    double length = sqrt(deltab.squaredNorm() + deltaw.squaredNorm());
    //cout << node_id_ << " IsStable: " << (length < epsilon_) << endl;
    return length < epsilon_;
  }
  double epsilon_;
  int max_iter_;
  double lr_;
  shared_ptr<Model> model_;
  shared_ptr<Reader> reader_;
#ifdef USE_MPI //TODO: Making these two variable global
  int node_id_;
  int nodes_nums_;
#endif
};

} // namespace paralleLR

#endif //PARALLELR_SOLVER_H
