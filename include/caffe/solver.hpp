#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>
#include <set>

#include "lazy-table-module.hpp"

#include "caffe/net.hpp"

namespace caffe {

struct PsConfig {
  bool no_ps;
  int worker_id;
  int num_workers;
  int slack;
  int batches_per_clock;
  LazyTableConfig lt_config;
  PsConfig() : no_ps(false), slack(0), batches_per_clock(1) {}
};

struct RowAccessInfo {
  vector<uint> row_ids;
  int num_vals;
  bool data_in_mem;  /* Volatile field only used at virtual iteration */
  int data_handle;  /* Volatile field only used at virtual iteration */
};

struct ParamInfo {
  int global_param_id;
  int val_offset;
};

struct ImbInfo {
  int global_imb_id;
  bool fetch;
  bool keep;
  ImbInfo(int g = -1, bool f = false, bool k = false) :
      global_imb_id(g), fetch(f), keep(k) {}
};

struct FetchKeep {
  bool fetch;
  bool keep;
  FetchKeep(bool f = false, bool k = false) : fetch(f), keep(k) {}
};

struct LayerHandles {
  int read_handle;
  int postread_handle;
  int bw_read_handle;
  int bw_postread_handle;
  int prewrite_handle;
  int write_handle;
  int history_access_handle;
  int history_postaccess_handle;
  vector<int> imbs_to_access_fw;
  vector<int> imbs_to_release_fw;
  vector<int> imb_diffs_to_access_fw;
  vector<int> imb_diffs_to_release_fw;
  vector<int> imbs_to_access_bw;
  vector<int> imbs_to_release_bw;
  vector<int> imb_diffs_to_access_bw;
  vector<int> imb_diffs_to_release_bw;
};

typedef std::map<int, FetchKeep> IntSet;
struct LayerInfo {
  int table_id;
  vector<uint> row_ids;
  vector<uint> history_data_row_ids;
  int num_vals;
  vector<ParamInfo> param_infos;
  IntSet imbs_used_fw;
  IntSet imb_diffs_used_fw;
  IntSet imbs_used_bw;
  IntSet imb_diffs_used_bw;
  vector<ImbInfo> imbs_to_access_fw;
  vector<ImbInfo> imbs_to_release_fw;
  vector<ImbInfo> imb_diffs_to_access_fw;
  vector<ImbInfo> imb_diffs_to_release_fw;
  vector<ImbInfo> imbs_to_access_bw;
  vector<ImbInfo> imbs_to_release_bw;
  vector<ImbInfo> imb_diffs_to_access_bw;
  vector<ImbInfo> imb_diffs_to_release_bw;
  int param_size;
  int imb_size;
  vector<LayerHandles> layer_handles;
  double fw_read_time;
  double fw_compute_time;
  double fw_write_time;
  double bw_read_time;
  double bw_compute_time;
  double bw_write_time;
};

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param, const PsConfig& ps_config,
      const Solver* root_solver = NULL);
  explicit Solver(const SolverParameter& param,
      const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  void InitPs();
  void SetPsParamValues();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  virtual Dtype ForwardBackwardUsingPs(const vector<Blob<Dtype>* > & bottom,
      const shared_ptr<Net<Dtype> >& net, bool test) = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;

  PsConfig ps_config_;
  vector<RowAccessInfo> imb_data_infos_;
  vector<RowAccessInfo> imb_diff_infos_;
  vector<LayerInfo> layer_infos_;
  vector<Blob<Dtype>*> test_net_output_blobs_;

  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  shared_ptr<LazyTableModule> ps_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  Dtype ForwardBackwardUsingPs(const vector<Blob<Dtype>* > & bottom,
      const shared_ptr<Net<Dtype> >& net, bool test) {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param, const PsConfig& ps_config)
      : Solver<Dtype>(param, ps_config) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ApplyUpdate();
  virtual Dtype ForwardBackwardUsingPs(const vector<Blob<Dtype>* > & bottom,
      const shared_ptr<Net<Dtype> >& net, bool test);
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(
      const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(
      const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

template <typename Dtype>
Solver<Dtype>* GetSolver(
    const SolverParameter& param, const PsConfig& ps_config) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param, ps_config);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

template <typename Dtype>
Solver<Dtype>* GetSolver(
    const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();
  PsConfig ps_config;
  ps_config.no_ps = true;

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_RMSPROP:
      return new RMSPropSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADADELTA:
      return new AdaDeltaSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADAM:
      return new AdamSolver<Dtype>(param, ps_config);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
