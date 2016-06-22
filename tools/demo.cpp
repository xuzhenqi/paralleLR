#include <memory>
#include <algorithm>
#include "io.h"
#include "model.h"
#include "solver.h"
#include "Eigen/Dense"
#include "matplotlibcpp.h"
#ifdef USE_MPI
#include "mpi.h"
#endif

namespace plt = matplotlibcpp;
using namespace std;
using namespace paralleLR;
using namespace Eigen;

void visual(const MatrixXd& data, const MatrixXd& label, const MatrixXd& w, 
            const MatrixXd& b) {
  double minx = data(0, 0), maxx = data(0, 0);
  for(int i = 0; i < data.rows(); ++i) {
    minx = min(data(i, 0), minx);
    maxx = max(data(i, 0), maxx);
    if (label(i, 0) > 0.5) {
      plt::plot({data(i, 0)}, {data(i, 1)}, ".r");
    } else {
      plt::plot({data(i, 0)}, {data(i, 1)}, ".g");
    }
  }
  plt::plot({minx, maxx}, {(-b(0, 0) - minx * w(0, 0)) / w(1, 0), 
            (-b(0, 0) - maxx * w(0, 0)) / w(1, 0)}, "b");
  plt::show();
}

int main(int argc, char* argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  shared_ptr<Reader> reader(new DummyReader(2000));
  shared_ptr<Model> model(new LRModel(2));
  SolverParameter param;
  param.max_iter_ = 10000;
  param.lr_ = 0.01;
  SGDSolver solver(param, reader, model);
  solver.Solve();
#ifdef USE_MPI
  int node_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
  DummyReader test(1000);
  cout << node_id << " Test Accuracy: " << model->Accuracy(test.Data(), test.Label()) <<
      endl;
  cout << node_id << " Learned weight: " << model->w() << " bias: " << model->b() << endl;
  if (node_id == 0) {
    visual(test.Data(), test.Label(), model->w(), model->b());
  }
  MPI_Finalize();
#else
  DummyReader test(1000);
  cout << "Test Accuracy: " << model->Accuracy(test.Data(), test.Label()) <<
      endl;
  visual(test.Data(), test.Label(), model->w(), model->b());
#endif
  return 0;
}
