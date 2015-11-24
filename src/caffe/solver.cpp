#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include <boost/make_shared.hpp>

#include <tbb/tick_count.h>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::cout;
using std::cerr;
using std::endl;
using boost::make_shared;

#define MULTI_TABLE
// #define LOCAL_DATA_IN_PS

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(
    const SolverParameter& param, const PsConfig& ps_config,
    const Solver* root_solver)
    : ps_config_(ps_config),
      net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(
    const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  ps_config_.no_ps = true;
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;

  /* Initialize parameter server */
  InitPs();
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <>
void Solver<float>::InitPs() {
  if (ps_config_.no_ps) {
    return;
  }

  vector<shared_ptr<Layer<float> > >& layers = this->net_->layers_;
  vector<string>& layer_types = this->net_->layer_types_;
  vector<bool>& layer_need_backward = this->net_->layer_need_backward_;
  vector<shared_ptr<Blob<float> > >& params = this->net_->params_;
  layer_infos_.resize(layers.size());
  int total_num_params = 0;
  int table_id = 0;
  int row_id = 0;
  int local_store_row_id = 0;
  int global_param_id = 0;

  /* Decide row keys for model parameters */
  // LOG(INFO) << "param sizes:";
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    shared_ptr<Layer<float> >& layer = layers[layer_id];
    LayerInfo& layer_info = layer_infos_[layer_id];
    int num_params = layer->blobs().size();
    if (num_params > 0) {
      layer_info.param_infos.resize(num_params);
      layer_info.table_id = table_id;
      total_num_params += num_params;
      layer_info.num_vals = 0;
      for (int param_id = 0; param_id < num_params; param_id++) {
        shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
        layer_info.param_infos[param_id].val_offset = layer_info.num_vals;
        layer_info.param_infos[param_id].global_param_id = global_param_id++;
        layer_info.num_vals += param->count();
      }
      int num_rows = (layer_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
      for (int i = 0; i < num_rows; i++) {
        layer_info.row_ids.push_back(row_id++);
        layer_info.history_data_row_ids.push_back(local_store_row_id++);
      }
#if defined(MULTI_TABLE)
      table_id++;
      row_id = 0;
#endif
    }
    layer_info.fw_read_time = 0;
    layer_info.fw_compute_time = 0;
    layer_info.fw_write_time = 0;
    layer_info.bw_read_time = 0;
    layer_info.bw_compute_time = 0;
    layer_info.bw_write_time = 0;
  }
  CHECK_EQ(total_num_params, params.size());
  int num_tables = row_id == 0 ? table_id : table_id + 1;

  /* Decide row keys for intermediate data blobs */
  vector<shared_ptr<Blob<float> > >& imbs = this->net_->blobs_;
  imb_data_infos_.resize(imbs.size());
  // LOG(INFO) << "imbs.size() = " << imbs.size();
  // LOG(INFO) << "imb sizes:" << imbs.size();
  for (int imb_id = 0; imb_id < imbs.size(); imb_id++) {
    RowAccessInfo& imb_info = imb_data_infos_[imb_id];
    imb_info.num_vals = imbs[imb_id]->count();
    cerr << "imbs[imb_id]->count() = " << imbs[imb_id]->count() << endl;
    int num_rows = (imb_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
    // cerr << num_rows << endl;
    for (int i = 0; i < num_rows; i++) {
      imb_info.row_ids.push_back(local_store_row_id++);
    }
    imb_info.data_in_mem = false;
    imb_info.data_handle = -1;
  }
  /* Decide row keys for intermediate diff blobs */
  imb_diff_infos_.resize(imbs.size());
  for (int imb_id = 0; imb_id < imbs.size(); imb_id++) {
    RowAccessInfo& imb_info = imb_diff_infos_[imb_id];
    imb_info.num_vals = imbs[imb_id]->count();
    int num_rows = (imb_info.num_vals + ROW_DATA_SIZE - 1) / ROW_DATA_SIZE;
    for (int i = 0; i < num_rows; i++) {
      imb_info.row_ids.push_back(local_store_row_id++);
    }
    imb_info.data_in_mem = false;
    imb_info.data_handle = -1;
  }

  /* Count total size of params and imbs */
  int input_size = 0;
  int imb_size = 0;
  int param_size = 0;
  int update_size = 0;
  for (int i = 0; i < imbs.size(); i++) {
    if (i < 2) {
      input_size += imbs[i]->count();
    } else {
      imb_size += imbs[i]->count();
    }
    /* Counting diffs */
    imb_size += imbs[i]->count();
  }
  for (int i = 0; i < params.size(); i++) {
    param_size += params[i]->count();
    update_size += params[i]->count();
    imb_size += params[i]->count();
  }
  cout << "Total sizes: " << endl;
  cout << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;

  /* Decide which intermediate blobs to access/release at each layer */
  vector<int>& net_output_blob_indices = this->net_->net_output_blob_indices_;
  IntSet net_output_set;
  for (int i = 0; i < net_output_blob_indices.size(); i++) {
    net_output_set[net_output_blob_indices[i]] = FetchKeep();
  }
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    vector<int>& bottom_imb_ids = this->net_->bottom_id_vecs_[layer_id];
    vector<int>& top_imb_ids = this->net_->top_id_vecs_[layer_id];
    IntSet& imbs_used_fw = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used_fw = layer_info.imb_diffs_used_fw;
    IntSet& imbs_used_bw = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used_bw = layer_info.imb_diffs_used_bw;
    for (int i = 0; i < bottom_imb_ids.size(); i++) {
      int blob_id = bottom_imb_ids[i];
      if (net_output_set.count(blob_id)) {
        /* Do not stream output blobs */
        continue;
      }
      /* Use (fetch, keep) all bottom data blobs in the forward pass */
      imbs_used_fw[blob_id] = FetchKeep(true, true);
      /* Use (fetch, no keep) all bottom data blobs in the backward pass,
       * except for data layers */
      if (layer_types[layer_id] != "Data") {
        imbs_used_bw[blob_id] = FetchKeep(true, false);
      }
      /* Use no bottom diff blobs in the forward pass */
      /* Use (no fetch, keep) all bottom diff blobs in the backward pass,
       * except for data layers */
      if (layer_types[layer_id] != "Data") {
        imb_diffs_used_bw[blob_id] = FetchKeep(false, true);
      }
    }
    for (int i = 0; i < top_imb_ids.size(); i++) {
      int blob_id = top_imb_ids[i];
      if (net_output_set.count(blob_id)) {
        /* Do not stream output blobs */
        continue;
      }
      /* Use (no fetch, keep) all top data blobs in the forward pass */
      imbs_used_fw[blob_id] = FetchKeep(false, true);
      /* Use (no fetch, keep) the top diff blobs only in loss layers
       * in the forward pass */
      if (layer_types[layer_id] == "SoftmaxWithLoss") {
        imb_diffs_used_fw[blob_id] = FetchKeep(false, true);
      }
      /* Use (fetch, no keep) the top data blobs only in ReLU, LRN, Pooling,
       * and SoftmaxWithLoss layers in the backward pass */
      if (layer_types[layer_id] == "ReLU" ||
          layer_types[layer_id] == "LRN" ||
          layer_types[layer_id] == "Pooling" ||
          layer_types[layer_id] == "SoftmaxWithLoss") {
        imbs_used_bw[blob_id] = FetchKeep(true, false);
      }
      /* Use (fetch, no keep) all top diff blobs in the backward pass,
       * except for data layers */
      if (layer_types[layer_id] != "Data") {
        imb_diffs_used_bw[blob_id] = FetchKeep(true, false);
      }
    }
    int total_count = 0;
    for (IntSet::iterator i = imbs_used_fw.begin();
         i != imbs_used_fw.end(); i++) {
      int imb_id = i->first;
      total_count += imbs[imb_id]->count();
    }
  }
  cout << "\nForwardbackward per layer sizes:" << endl;
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    IntSet& imbs_used = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_fw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }
  for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    IntSet& imbs_used = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_bw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    update_size += layer_info.num_vals;
    imb_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }
  cout << "\nForwardbackward two layer sizes:" << endl;
  for (int layer_id = 0; layer_id < layers.size() - 1; layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    IntSet& imbs_used = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_fw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    LayerInfo& next_layer_info = layer_infos_[layer_id + 1];
    IntSet& next_imbs_used = next_layer_info.imbs_used_fw;
    IntSet& next_imb_diffs_used = next_layer_info.imb_diffs_used_fw;
    for (IntSet::iterator i = next_imbs_used.begin();
         i != next_imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used.count(imb_id)) {
        if (imb_id < 2) {
          input_size += imbs[imb_id]->count();
        } else {
          imb_size += imbs[imb_id]->count();
        }
      }
    }
    for (IntSet::iterator i = next_imb_diffs_used.begin();
         i != next_imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used.count(imb_id)) {
        imb_size += imbs[imb_id]->count();
      }
    }
    param_size += next_layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }
  {
    /* For the last layer */
    int layer_id = layer_infos_.size() - 1;
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    IntSet& imbs_used = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_bw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    update_size += layer_info.num_vals;
    imb_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }
  for (int layer_id = layer_infos_.size() - 1; layer_id >= 1; layer_id--) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    IntSet& imbs_used = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_bw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals * 3;
    LayerInfo& next_layer_info = layer_infos_[layer_id - 1];
    IntSet& next_imbs_used = next_layer_info.imbs_used_bw;
    IntSet& next_imb_diffs_used = next_layer_info.imb_diffs_used_bw;
    for (IntSet::iterator i = next_imbs_used.begin();
         i != next_imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used.count(imb_id)) {
        if (imb_id < 2) {
          input_size += imbs[imb_id]->count();
        } else {
          imb_size += imbs[imb_id]->count();
        }
      }
    }
    for (IntSet::iterator i = next_imb_diffs_used.begin();
         i != next_imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used.count(imb_id)) {
        imb_size += imbs[imb_id]->count();
      }
    }
    param_size += layer_info.num_vals;
    update_size += layer_info.num_vals;
    imb_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }

  /* Decide imbs to accesss/release in forward pass */
  for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    IntSet& imbs_used = layer_info.imbs_used_fw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_fw;
    /* Decide imbs to access in forward pass */
    vector<ImbInfo>& imbs_to_access = layer_info.imbs_to_access_fw;
    vector<ImbInfo>& imb_diffs_to_access = layer_info.imb_diffs_to_access_fw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_data_infos_[imb_id].data_in_mem) {
        imb_data_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imbs_to_access.push_back(imb_info);
      }
    }
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diff_infos_[imb_id].data_in_mem) {
        imb_diff_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imb_diffs_to_access.push_back(imb_info);
      }
    }
    /* Decide imbs to release in forward pass */
    vector<ImbInfo>& imbs_to_release = layer_info.imbs_to_release_fw;
    vector<ImbInfo>& imb_diffs_to_release = layer_info.imb_diffs_to_release_fw;
    /* Release the blobs that are not used in the next layer */
    IntSet& imbs_used_next_layer = layer_id < layers.size() - 1 ?
        layer_infos_[layer_id + 1].imbs_used_fw :
        layer_infos_[layer_id].imbs_used_bw;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used_next_layer.count(imb_id)) {
        CHECK(imb_data_infos_[imb_id].data_in_mem);
        imb_data_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imbs_to_release.push_back(imb_info);
      }
    }
    IntSet& imb_diffs_used_next_layer = layer_id < layers.size() - 1 ?
        layer_infos_[layer_id + 1].imb_diffs_used_fw :
        layer_infos_[layer_id].imb_diffs_used_bw;
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used_next_layer.count(imb_id)) {
        CHECK(imb_diff_infos_[imb_id].data_in_mem);
        imb_diff_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imb_diffs_to_release.push_back(imb_info);
      }
    }
  }
  // /* Decide the last backward layer.
   // * We assume all layers above it need backward.
   // * Actually data layer is the only one that I think doesn't need backward. */
  // /* Decide imbs to accesss/release in backward pass */
  // int last_layer_needs_backward = -1;
  // for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
    // if (layer_need_backward[layer_id]) {
      // last_layer_needs_backward = layer_id;
    // }
  // }
  for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
    if (!layer_need_backward[layer_id]) {
      /* We assume only the data layer doesn't need backward */
      // LOG(INFO) << "layer " << layer_id << " doesn't need backward";
      continue;
    }
    LayerInfo& layer_info = layer_infos_[layer_id];
    IntSet& imbs_used = layer_info.imbs_used_bw;
    IntSet& imb_diffs_used = layer_info.imb_diffs_used_bw;
    vector<ImbInfo>& imbs_to_access = layer_info.imbs_to_access_bw;
    vector<ImbInfo>& imb_diffs_to_access = layer_info.imb_diffs_to_access_bw;
    /* Decide imbs to access in backward pass */
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_data_infos_[imb_id].data_in_mem) {
        imb_data_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imbs_to_access.push_back(imb_info);
      }
    }
    /* Decide imb diffs to access in backward pass */
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diff_infos_[imb_id].data_in_mem) {
        imb_diff_infos_[imb_id].data_in_mem = true;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.fetch = i->second.fetch;
        imb_diffs_to_access.push_back(imb_info);
      }
    }
    /* Decide imbs to release in backward pass */
    vector<ImbInfo>& imbs_to_release = layer_info.imbs_to_release_bw;
    vector<ImbInfo>& imb_diffs_to_release = layer_info.imb_diffs_to_release_bw;
    IntSet empty_set;
    IntSet *imbs_used_next_layer_ptr = &empty_set;
    int next_layer_id = layer_id - 1;
    while (next_layer_id >= 0) {
      if (layer_need_backward[next_layer_id]) {
        imbs_used_next_layer_ptr =
            &layer_infos_[next_layer_id].imbs_used_bw;
        break;
      }
      next_layer_id--;
    }
    IntSet& imbs_used_next_layer = *imbs_used_next_layer_ptr;
    for (IntSet::iterator i = imbs_used.begin(); i != imbs_used.end(); i++) {
      int imb_id = i->first;
      if (!imbs_used_next_layer.count(imb_id)) {
        CHECK(imb_data_infos_[imb_id].data_in_mem);
        imb_data_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imbs_to_release.push_back(imb_info);
      }
    }
    IntSet *imb_diffs_used_next_layer_ptr = &empty_set;
    next_layer_id = layer_id - 1;
    while (next_layer_id >= 0) {
      if (layer_need_backward[next_layer_id]) {
        imb_diffs_used_next_layer_ptr =
            &layer_infos_[next_layer_id].imb_diffs_used_bw;
        break;
      }
      next_layer_id--;
    }
    IntSet& imb_diffs_used_next_layer = *imb_diffs_used_next_layer_ptr;
    for (IntSet::iterator i = imb_diffs_used.begin();
         i != imb_diffs_used.end(); i++) {
      int imb_id = i->first;
      if (!imb_diffs_used_next_layer.count(imb_id)) {
        CHECK(imb_diff_infos_[imb_id].data_in_mem);
        imb_diff_infos_[imb_id].data_in_mem = false;
        ImbInfo imb_info;
        imb_info.global_imb_id = imb_id;
        imb_info.keep = i->second.keep;
        imb_diffs_to_release.push_back(imb_info);
      }
    }
  }
  /* All blobs should have been released */
  for (int i = 0; i < imb_data_infos_.size(); i++) {
    CHECK(!imb_data_infos_[i].data_in_mem) << "i = " << i;
  }
  for (int i = 0; i < imb_diff_infos_.size(); i++) {
    CHECK(!imb_diff_infos_[i].data_in_mem) << "i = " << i;
  }

  /* Print the size of imbs that need to be fetched */
  cout << "\nSize of imbs that need to be fetched during forwardbackward:" << endl;
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    for (int i = 0; i < layer_info.imbs_to_access_fw.size(); i++) {
      ImbInfo& imb_info = layer_info.imbs_to_access_fw[i];
      int imb_id = imb_info.global_imb_id;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    /* Access intermediate diff blobs */
    for (int i = 0; i < layer_info.imb_diffs_to_access_fw.size(); i++) {
      ImbInfo& imb_info = layer_info.imb_diffs_to_access_fw[i];
      int imb_id = imb_info.global_imb_id;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }
  for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    int input_size = 0;
    int imb_size = 0;
    int param_size = 0;
    int update_size = 0;
    for (int i = 0; i < layer_info.imbs_to_access_bw.size(); i++) {
      ImbInfo& imb_info = layer_info.imbs_to_access_bw[i];
      int imb_id = imb_info.global_imb_id;
      if (imb_id < 2) {
        input_size += imbs[imb_id]->count();
      } else {
        imb_size += imbs[imb_id]->count();
      }
    }
    /* Access intermediate diff blobs */
    for (int i = 0; i < layer_info.imb_diffs_to_access_bw.size(); i++) {
      ImbInfo& imb_info = layer_info.imb_diffs_to_access_bw[i];
      int imb_id = imb_info.global_imb_id;
      imb_size += imbs[imb_id]->count();
    }
    param_size += layer_info.num_vals;
    update_size += layer_info.num_vals;
    imb_size += layer_info.num_vals;
    cout << layer_id << ',' << input_size << ',' << imb_size
         << ',' << param_size << ',' << update_size << endl;
  }

  int64_t total_size = 0;
  int64_t read_size = 0;
  int64_t write_size = 0;
  for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
    LayerInfo& layer_info = layer_infos_[layer_id];
    layer_info.layer_handles.resize(ps_config_.batches_per_clock);
    for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
      layer_handles.imbs_to_access_fw.resize(layer_info.imbs_to_access_fw.size());
      layer_handles.imbs_to_release_fw.resize(layer_info.imbs_to_release_fw.size());
      layer_handles.imb_diffs_to_access_fw.resize(layer_info.imb_diffs_to_access_fw.size());
      layer_handles.imb_diffs_to_release_fw.resize(layer_info.imb_diffs_to_release_fw.size());
      layer_handles.imbs_to_access_bw.resize(layer_info.imbs_to_access_bw.size());
      layer_handles.imbs_to_release_bw.resize(layer_info.imbs_to_release_bw.size());
      layer_handles.imb_diffs_to_access_bw.resize(layer_info.imb_diffs_to_access_bw.size());
      layer_handles.imb_diffs_to_release_bw.resize(layer_info.imb_diffs_to_release_bw.size());
    }
  }


  /* Initialize LazyTable */
  ps_config_.lt_config.num_tables = num_tables;
  ps_ = make_shared<LazyTableModule>(
      ps_config_.worker_id, ps_config_.lt_config);
  ps_->thread_start();

  /* Virtual iteration */ 
  for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
    /* Virtual iteration, forward pass */
    for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_fw[i];
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imbs_to_access_fw.size());
        int& handle = layer_handles.imbs_to_access_fw[i];
        handle = ps_->virtual_localaccess_batch(
            access_info.row_ids, access_info.num_vals, imb_info.fetch);
        access_info.data_handle = handle;
        total_size += access_info.num_vals;
        read_size += imb_info.fetch ? access_info.num_vals : 0;
        CHECK_GE(read_size, 0);
      }
      /* Access intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_fw[i];
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_fw.size());
        int& handle = layer_handles.imb_diffs_to_access_fw[i];
        handle = ps_->virtual_localaccess_batch(
            access_info.row_ids, access_info.num_vals, imb_info.fetch);
        access_info.data_handle = handle;
        total_size += access_info.num_vals;
        read_size += imb_info.fetch ? access_info.num_vals : 0;
        CHECK_GE(read_size, 0);
      }
#endif
      /* Read model parameters */
      if (layer_info.param_infos.size()) {
        layer_handles.read_handle = ps_->virtual_read_batch(
            layer_info.table_id, layer_info.row_ids,
            ps_config_.slack, layer_info.num_vals);
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_fw[i];
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imbs_to_release_fw.size());
        int& handle = layer_handles.imbs_to_release_fw[i];
        handle = ps_->virtual_postlocalaccess_batch(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
        write_size += imb_info.keep ? access_info.num_vals : 0;
        CHECK_GE(write_size, 0);
      }
      /* Release intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_fw[i];
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imb_diffs_to_release_fw.size());
        int& handle = layer_handles.imb_diffs_to_release_fw[i];
        handle = ps_->virtual_postlocalaccess_batch(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
        write_size += imb_info.keep ? access_info.num_vals : 0;
        CHECK_GE(write_size, 0);
      }
#endif
      /* Release model parameters */
      if (layer_info.param_infos.size()) {
        layer_handles.postread_handle = ps_->virtual_postread_batch(
            layer_handles.read_handle);
      }
    }
    /* Virtual iteration, backward pass */
    for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
      if (!layer_need_backward[layer_id]) {
        /* We assume only the data layer doesn't need backward */
        continue;
      }
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];
#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_data_infos_.size());
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imbs_to_access_bw.size());
        int& handle = layer_handles.imbs_to_access_bw[i];
        handle = ps_->virtual_localaccess_batch(
            access_info.row_ids, access_info.num_vals, imb_info.fetch);
        access_info.data_handle = handle;
        total_size += access_info.num_vals;
        read_size += imb_info.fetch ? access_info.num_vals : 0;
        CHECK_GE(read_size, 0);
      }
      /* Access intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_diff_infos_.size());
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_bw.size());
        int& handle = layer_handles.imb_diffs_to_access_bw[i];
        handle = ps_->virtual_localaccess_batch(
            access_info.row_ids, access_info.num_vals, imb_info.fetch);
        access_info.data_handle = handle;
        total_size += access_info.num_vals;
        read_size += imb_info.fetch ? access_info.num_vals : 0;
        CHECK_GE(read_size, 0);
      }
#endif
      /* Read and prewrite model parameters */
      if (layer_info.param_infos.size()) {
        layer_handles.prewrite_handle = ps_->virtual_prewrite_batch(
            layer_info.table_id, layer_info.row_ids, layer_info.num_vals);
        layer_handles.bw_read_handle = ps_->virtual_read_batch(
            layer_info.table_id, layer_info.row_ids,
            ps_config_.slack, layer_info.num_vals);
        layer_handles.history_access_handle =
            ps_->virtual_localaccess_batch(
                layer_info.history_data_row_ids, layer_info.num_vals,
                /* fetch */ true);
      }
#if defined(LOCAL_DATA_IN_PS)
      /* Postaccess intermediate data blobs */
      for (int i = 0; i < layer_info.imbs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_data_infos_.size());
        RowAccessInfo& access_info = imb_data_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imbs_to_release_bw.size());
        int& handle = layer_handles.imbs_to_release_bw[i];
        handle = ps_->virtual_postlocalaccess_batch(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
        write_size += imb_info.keep ? access_info.num_vals : 0;
        CHECK_GE(write_size, 0);
      }
      /* Postaccess intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_bw[i];
        CHECK_LT(imb_info.global_imb_id, imb_diff_infos_.size());
        RowAccessInfo& access_info = imb_diff_infos_[imb_info.global_imb_id];
        CHECK_GE(access_info.data_handle, 0);
        CHECK_LT(i, layer_handles.imb_diffs_to_release_bw.size());
        int& handle = layer_handles.imb_diffs_to_release_bw[i];
        handle = ps_->virtual_postlocalaccess_batch(
            access_info.data_handle, imb_info.keep);
        access_info.data_handle = -1;
        write_size += imb_info.keep ? access_info.num_vals : 0;
        CHECK_GE(write_size, 0);
      }
#endif
      /* Postread and write model parameters */
      if (layer_info.param_infos.size()) {
        layer_handles.write_handle = ps_->virtual_write_batch(
            layer_handles.prewrite_handle);
        layer_handles.bw_postread_handle = ps_->virtual_postread_batch(
            layer_handles.bw_read_handle);
        layer_handles.history_postaccess_handle =
            ps_->virtual_postlocalaccess_batch(
                layer_handles.history_access_handle, /* keep */ true);
      }
    }
  }
  ps_->virtual_clock();
  ps_->finish_virtual_iteration();
  LOG(INFO) << "Virtual iteration done";
  cout << "total_size = " << total_size << endl;
  cout << "read_size = " << read_size << endl;
  cout << "write_size = " << write_size << endl;

  /* Set initial parameter values */
  if (ps_config_.worker_id == 0) {
    for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
      shared_ptr<Layer<float> >& layer = layers[layer_id];
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[0];
      if (layer_info.param_infos.size()) {
        /* Pre-write */
        RowOpVal *inc_buffer;
        ps_->preinc_batch(&inc_buffer, layer_handles.prewrite_handle);
        float *params_vals = reinterpret_cast<float *>(inc_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(param_vals, false);
          /* "false" means that we don't change head here,
           * because we want to keep what's currently in CPU memory */
        }
      }
      /* Let the layer initialize values */
      layer->InitializeValues();
      if (layer_info.param_infos.size()) {
        /* Write */
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          /* Values are filled in CPU memory, do a gpu_data() call to copy them
           * to GPU memory */
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->gpu_data();
          // const float *param_data = param->gpu_data();
          // float param_dot;
          // caffe_gpu_dot<float>(param->count(), param_data, param_data, &param_dot);
          // LOG(INFO) << "param_dot = " << param_dot;
          param->set_gpu_data(NULL, true);
          /* "true" means that we don't keep CPU data */
        }
        ps_->inc_batch(layer_handles.write_handle);
      }
    }
  }
  LOG(INFO) << "Set initial parameter values done";
  ps_->iterate();
  ps_->start_opseq();
  LOG(INFO) << "opseq started";
}

template <>
float SGDSolver<float>::ForwardBackwardUsingPs(
    const vector<Blob<float>* >& bottom,
    const shared_ptr<Net<float> >& net, bool test) {
  vector<shared_ptr<Layer<float> > >& layers = net->layers_;
  vector<vector<Blob<float>*> >& bottom_vecs = net->bottom_vecs_;
  vector<vector<Blob<float>*> >& top_vecs = net->top_vecs_;
  vector<bool>& layer_need_backward = net->layer_need_backward_;
  vector<vector<bool> >& bottom_need_backward = net->bottom_need_backward_;
  vector<shared_ptr<Blob<float> > >& imbs = net->blobs_;
  vector<string>& layer_names = net->layer_names_;
  tbb::tick_count tick_start;

  // if (test) {
    // LOG(INFO) << "TEST";
  // } else {
    // LOG(INFO) << "TRAIN";
  // }

  /* Forward */
  // LOG(INFO) << "Forward";
  float loss = 0;
  for (int batch_id = 0; batch_id < ps_config_.batches_per_clock; batch_id++) {
    for (int layer_id = 0; layer_id < layer_infos_.size(); layer_id++) {
      // LOG(INFO) << "Layer " << layer_id << ": " << layer_names[layer_id];
      CHECK_LT(layer_id, layers.size());
      shared_ptr<Layer<float> >& layer = layers[layer_id];
      CHECK(layer);
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];

#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      tick_start = tbb::tick_count::now();
      // LOG(INFO) << "Read intermediate data blobs";
      for (int i = 0; i < layer_info.imbs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_fw[i];
        CHECK_LT(i, layer_handles.imbs_to_access_fw.size());
        int handle = layer_handles.imbs_to_access_fw[i];
        // LOG(INFO) << "Read data " << imb_info.global_imb_id;
        CHECK_LT(imb_info.global_imb_id, imbs.size());
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowOpVal *read_buffer;
        ps_->localaccess_batch(&read_buffer, handle);
        CHECK(!imb->check_gpu_data())
            << "layer " << layer_names[layer_id] << " has gpu data "
            << imb_info.global_imb_id;
        imb->set_gpu_data(reinterpret_cast<float *>(read_buffer), true);
      }
      /* Access intermediate diff blobs */
      // LOG(INFO) << "Read intermediate diff blobs";
      for (int i = 0; i < layer_info.imb_diffs_to_access_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_fw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_fw.size());
        int handle = layer_handles.imb_diffs_to_access_fw[i];
        // LOG(INFO) << "Read data " << imb_info.global_imb_id;
        CHECK_LT(imb_info.global_imb_id, imbs.size());
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowOpVal *read_buffer;
        ps_->localaccess_batch(&read_buffer, handle);
        // LOG(INFO) << "buffer = " << read_buffer;
        CHECK(!imb->check_gpu_diff())
            << "layer " << layer_names[layer_id] << " has gpu diff";
        imb->set_gpu_diff(reinterpret_cast<float *>(read_buffer), true);
      }
#endif
      /* Read model parameters */
      if (layer_info.param_infos.size()) {
        // LOG(INFO) << "Read params";
        RowOpVal *read_buffer;
        ps_->read_batch(&read_buffer, layer_handles.read_handle);
        float *params_vals = reinterpret_cast<float *>(read_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          CHECK(!param->check_gpu_data())
              << "layer " << layer_names[layer_id] << " has gpu param";
          param->set_gpu_data(param_vals, true);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.fw_read_time += (tbb::tick_count::now() - tick_start).seconds();
      }

      /* Forward calculation */
      // LOG(INFO) << "Forward calculation";
      tick_start = tbb::tick_count::now();
      float layer_loss =
          layer->Forward(bottom_vecs[layer_id], top_vecs[layer_id]);
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      // LOG(INFO) << "layer_loss = " << layer_loss;
      loss += layer_loss;
      if (!test) {
        layer_info.fw_compute_time += (tbb::tick_count::now() - tick_start).seconds();
      }

#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      tick_start = tbb::tick_count::now();
      // LOG(INFO) << "Release intermediate data blobs";
      for (int i = 0; i < layer_info.imbs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_fw[i];
        CHECK_LT(i, layer_handles.imbs_to_release_fw.size());
        int handle = layer_handles.imbs_to_release_fw[i];
        // LOG(INFO) << "Release data " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_data();
           /* Make sure everything is copied to GPU memory */
        imb->set_gpu_data(NULL, true);
        ps_->postlocalaccess_batch(handle);
      }
      /* Release intermediate diff blobs */
      // LOG(INFO) << "Release intermediate diff blobs";
      for (int i = 0; i < layer_info.imb_diffs_to_release_fw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_fw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_release_fw.size());
        int handle = layer_handles.imb_diffs_to_release_fw[i];
        // LOG(INFO) << "Release data " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_diff();
           /* Make sure everything is copied to GPU memory */
        imb->set_gpu_diff(NULL, true);
        ps_->postlocalaccess_batch(handle);
      }
#endif
      /* Release read buffers */
      if (layer_info.param_infos.size()) {
        // LOG(INFO) << "Release read buffers";
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(NULL, true);
        }
        ps_->postread_batch(layer_handles.postread_handle);
      }
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.fw_write_time += (tbb::tick_count::now() - tick_start).seconds();
      }
    }

    /* Backward */
    // LOG(INFO) << "Backward";
    for (int layer_id = layer_infos_.size() - 1; layer_id >= 0; layer_id--) {
      // LOG(INFO) << "Layer " << layer_id << ": " << layer_names[layer_id];
      CHECK_LT(layer_id, layer_need_backward.size());
      if (!test && !layer_need_backward[layer_id]) {
        continue;
      }
      CHECK_LT(layer_id, layers.size());
      shared_ptr<Layer<float> >& layer = layers[layer_id];
      CHECK(layer);
      LayerInfo& layer_info = layer_infos_[layer_id];
      LayerHandles& layer_handles = layer_info.layer_handles[batch_id];

#if defined(LOCAL_DATA_IN_PS)
      /* Access intermediate data blobs */
      tick_start = tbb::tick_count::now();
      for (int i = 0; i < layer_info.imbs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_access_bw[i];
        CHECK_LT(i, layer_handles.imbs_to_access_bw.size());
        int handle = layer_handles.imbs_to_access_bw[i];
        // LOG(INFO) << "Read data " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowOpVal *imb_buffer;
        ps_->localaccess_batch(&imb_buffer, handle);
        CHECK(!imb->check_gpu_data())
            << "layer " << layer_names[layer_id] << " has gpu data";
        imb->set_gpu_data(reinterpret_cast<float *>(imb_buffer), true);
      }
      /* Access intermediate diff blobs */
      for (int i = 0; i < layer_info.imb_diffs_to_access_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_access_bw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_access_bw.size());
        int handle = layer_handles.imb_diffs_to_access_bw[i];
        // LOG(INFO) << "Read diff " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        RowOpVal *imb_buffer;
        ps_->localaccess_batch(&imb_buffer, handle);
        CHECK(!imb->check_gpu_diff())
            << "layer " << layer_names[layer_id] << " has gpu diff";
        imb->set_gpu_diff(reinterpret_cast<float *>(imb_buffer), true);
      }
#endif
      if (layer_info.param_infos.size()) {
        /* Prepare write buffers */
        // LOG(INFO) << "Prepare write buffers";
        RowOpVal *write_buffer;
        ps_->preinc_batch(&write_buffer, layer_handles.prewrite_handle);
        float *write_params_vals = reinterpret_cast<float *>(write_buffer);
        size_t size = layer_info.num_vals * sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(
            write_params_vals, 0, size, Caffe::cuda_stream()));
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &write_params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_diff(param_vals, true);
          /* "true" means that we don't keep CPU data */
        }
        /* Read params */
        // LOG(INFO) << "Read params";
        RowOpVal *read_buffer;
        ps_->read_batch(&read_buffer, layer_handles.bw_read_handle);
        float *read_params_vals = reinterpret_cast<float *>(read_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *param_vals = &read_params_vals[param_val_offset];
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(param_vals, true);
        }
        /* Access local updates history */
        RowOpVal *history_buffer;
        ps_->localaccess_batch(&history_buffer, layer_handles.history_access_handle);
        float *history_vals = reinterpret_cast<float *>(history_buffer);
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int param_val_offset = layer_info.param_infos[param_id].val_offset;
          float *history_param_vals = &history_vals[param_val_offset];
          int global_param_id = layer_info.param_infos[param_id].global_param_id;
          history_[global_param_id]->set_gpu_data(history_param_vals, true);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.bw_read_time += (tbb::tick_count::now() - tick_start).seconds();
      }

      if (!test) {
        /* Backward calculation */
        // LOG(INFO) << "Backward calculation";
        tick_start = tbb::tick_count::now();
        layer->Backward(top_vecs[layer_id], bottom_need_backward[layer_id],
            bottom_vecs[layer_id]);
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        /* Compute diff */
        // LOG(INFO) << "Compute diff";
        layer->ComputeDiff(top_vecs[layer_id], bottom_need_backward[layer_id],
            bottom_vecs[layer_id]);
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        layer_info.bw_compute_time += (tbb::tick_count::now() - tick_start).seconds();
      }

#if defined(LOCAL_DATA_IN_PS)
      /* Release intermediate data blobs */
      // LOG(INFO) << "Release intermediate data blobs";
      tick_start = tbb::tick_count::now();
      for (int i = 0; i < layer_info.imbs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imbs_to_release_bw[i];
        CHECK_LT(i, layer_handles.imbs_to_release_bw.size());
        int handle = layer_handles.imbs_to_release_bw[i];
        // LOG(INFO) << "Release data " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_data();
          /* Make sure everything is copied to GPU memory */
        imb->set_gpu_data(NULL, true);
        ps_->postlocalaccess_batch(handle);
      }
      /* Release intermediate diff blobs */
      // LOG(INFO) << "Release intermediate diff blobs";
      for (int i = 0; i < layer_info.imb_diffs_to_release_bw.size(); i++) {
        ImbInfo& imb_info = layer_info.imb_diffs_to_release_bw[i];
        CHECK_LT(i, layer_handles.imb_diffs_to_release_bw.size());
        int handle = layer_handles.imb_diffs_to_release_bw[i];
        // LOG(INFO) << "Release diff " << imb_info.global_imb_id;
        shared_ptr<Blob<float> >& imb = imbs[imb_info.global_imb_id];
        imb->gpu_diff();
          /* Make sure everything is copied to GPU memory */
        imb->set_gpu_diff(NULL, true);
        ps_->postlocalaccess_batch(handle);
      }
#endif
      CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
      if (!test) {
        layer_info.bw_write_time += (tbb::tick_count::now() - tick_start).seconds();
      }

      if (layer_info.param_infos.size()) {
        // LOG(INFO) << "Finish writing";
        tick_start = tbb::tick_count::now();
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int global_param_id = layer_info.param_infos[param_id].global_param_id;
          if (!test) {
            /* Adjust gradient */
            float rate = GetLearningRate();
            // Normalize(global_param_id);
            Regularize(global_param_id);
            // LOG(INFO) << "ComputeUpdateValue";
            ComputeUpdateValue(global_param_id, rate);
            CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
          }
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->gpu_diff();
            /* Make sure everything is copied to GPU memory */
          param->set_gpu_diff(NULL, true);
        }
        if (!test) {
          layer_info.bw_compute_time += (tbb::tick_count::now() - tick_start).seconds();
        }

        tick_start = tbb::tick_count::now();
        /* Apply updates to PS */
        ps_->inc_batch(layer_handles.write_handle);
        /* Release read buffers */
        // LOG(INFO) << "Release read buffers";
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          shared_ptr<Blob<float> >& param = layer->blobs()[param_id];
          param->set_gpu_data(NULL, true);
        }
        ps_->postread_batch(layer_handles.bw_postread_handle);
        /* Release local updates history */
        for (int param_id = 0;
            param_id < layer_info.param_infos.size(); param_id++) {
          int global_param_id = layer_info.param_infos[param_id].global_param_id;
          history_[global_param_id]->gpu_data();
            /* Make sure everything is copied to GPU memory */
          history_[global_param_id]->set_gpu_data(NULL, true);
        }
        ps_->postlocalaccess_batch(layer_handles.history_postaccess_handle);
        CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
        if (!test) {
          layer_info.bw_write_time += (tbb::tick_count::now() - tick_start).seconds();
        }
      }
    }
  }
  ps_->iterate();
  loss /= ps_config_.batches_per_clock;
  return loss;
}

template <>
void Solver<double>::InitPs() {
  CHECK(0);
}

template <>
double SGDSolver<double>::ForwardBackwardUsingPs(
    const vector<Blob<double>* > & bottom,
    const shared_ptr<Net<double> >& net, bool test) {
  CHECK(0);
  return 0;
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  double read_ps_time = 0.0;
  double compute_time = 0.0;
  double adjust_update_time = 0.0;
  double update_ps_time = 0.0;
  tbb::tick_count tick_start = tbb::tick_count::now();

  while (iter_ < stop_iter) {
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    tbb::tick_count compute_start = tbb::tick_count::now();
    Dtype loss = 0;
    CHECK_EQ(param_.iter_size(), 1);
    loss = ForwardBackwardUsingPs(bottom_vec, this->net_, /* train */ false);
    CUDA_CHECK(cudaStreamSynchronize(Caffe::cuda_stream()));
    compute_time += (tbb::tick_count::now() - compute_start).seconds();
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG_IF(INFO, Caffe::root_solver())
          << "Iteration " << iter_ << ", loss = " << smoothed_loss
          << " worker" << ps_config_.worker_id;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }

    if (iter_ % 1000 == 0 || iter_ == stop_iter) {
      double training_time = (tbb::tick_count::now() - tick_start).seconds();
      double read_time = 0;
      double write_time = 0;
      double compute_time = 0;
      for (int i = 0; i < layer_infos_.size(); i++) {
        read_time += layer_infos_[i].fw_read_time;
        read_time += layer_infos_[i].bw_read_time;
        write_time += layer_infos_[i].fw_write_time;
        write_time += layer_infos_[i].bw_write_time;
        compute_time += layer_infos_[i].fw_compute_time;
        compute_time += layer_infos_[i].bw_compute_time;
      }
      LOG(INFO) << "Read PS time: " << read_time;
      LOG(INFO) << "Write PS time: " << write_time;
      LOG(INFO) << "Compute time: " << compute_time;
      // LOG(INFO) << "Compute time: " << training_time - read_time;
      LOG(INFO) << "Training time: " << training_time;
      // LOG(INFO) << "Per layer forwardbackward times:";
      // for (int i = 0; i < layer_infos_.size(); i++) {
        // cerr << i << "," << layer_infos_[i].fw_read_time
             // << "," << layer_infos_[i].fw_compute_time
             // << "," << layer_infos_[i].fw_write_time
             // << endl;
      // }
      // for (int i = layer_infos_.size() - 1; i >= 0; i--) {
        // cerr << i << "," << layer_infos_[i].bw_read_time
             // << "," << layer_infos_[i].bw_compute_time
             // << "," << layer_infos_[i].bw_write_time
             // << endl;
      // }
    }
  }
  if (!ps_config_.no_ps) {
    string json_stats = ps_->json_stats();
    cerr << json_stats << endl;
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  // if (param_.display() && iter_ % param_.display() == 0) {
    // Dtype loss;
    // net_->ForwardPrefilled(&loss);
    // LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  // }
  // if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    // TestAll();
  // }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss =
        ForwardBackwardUsingPs(bottom_vec, test_net, /* test */ true);
    const vector<Blob<Dtype>*>& result = test_net->net_output_blobs_;
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      model_filename = SnapshotToBinaryProto();
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      model_filename = SnapshotToHDF5();
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  string filename(param_.snapshot_prefix());
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  return filename + iter_str_buffer + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    /* Cui: TODO: do we really need these? */
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  // if (!this->ps_config_.no_ps) {
    // /* Cui: now we have more workers */
    // rate /= this->ps_config_.num_workers;
  // }

  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  if (this->ps_config_.no_ps) {
    this->net_->Update();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        CHECK(0);
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  // Cui: I made the local learning rate negative, so that the updates will be
  // added to the parameter data instead of subtracted
  // Dtype local_rate = rate * net_params_lr[param_id];
  Dtype local_rate = -rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              history_[param_id]->mutable_gpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file" << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  CHECK(Caffe::root_solver());
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              this->history_[param_id]->mutable_cpu_data());

    // compute update: step back then over step
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->cpu_data(), -momentum,
        this->update_[param_id]->mutable_cpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              this->history_[param_id]->mutable_gpu_data());

    // compute update: step back then over step
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->gpu_data(), -momentum,
        this->update_[param_id]->mutable_gpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  CHECK(Caffe::root_solver());
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype delta = this->param_.delta();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_add(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        this->history_[param_id]->cpu_data(),
        this->history_[param_id]->mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
              this->history_[param_id]->cpu_data(), Dtype(0.5),
              this->update_[param_id]->mutable_cpu_data());

    caffe_add_scalar(net_params[param_id]->count(),
              delta, this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
              net_params[param_id]->cpu_diff(),
              this->update_[param_id]->cpu_data(),
              this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->cpu_data(), Dtype(0),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // compute square of gradient in update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_add(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        this->history_[param_id]->gpu_data(),
        this->history_[param_id]->mutable_gpu_data());

    // prepare update
    caffe_gpu_powx(net_params[param_id]->count(),
              this->history_[param_id]->gpu_data(), Dtype(0.5),
              this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_add_scalar(net_params[param_id]->count(),
              delta, this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_div(net_params[param_id]->count(),
              net_params[param_id]->gpu_diff(),
              this->update_[param_id]->gpu_data(),
              this->update_[param_id]->mutable_gpu_data());

    // scale and copy
    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->gpu_data(), Dtype(0),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  Dtype rms_decay = this->param_.rms_decay();
  Dtype local_rate = rate * net_params_lr[param_id];

  switch (Caffe::mode()) {
  case Caffe::CPU:
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id] -> count(),
        Dtype(1-rms_decay), this->update_[param_id]->cpu_data(),
        rms_decay, this->history_[param_id]-> mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    caffe_add_scalar(net_params[param_id]->count(),
        delta, this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), this->update_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->cpu_data(), Dtype(0),
        net_params[param_id]->mutable_cpu_diff());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    // compute square of gradient in update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history
    caffe_gpu_axpby(net_params[param_id] -> count(),
        Dtype(1-rms_decay), this->update_[param_id]->gpu_data(),
        rms_decay, this->history_[param_id]-> mutable_gpu_data());

    // prepare update
    caffe_gpu_powx(net_params[param_id]->count(),
        this->history_[param_id]->gpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_add_scalar(net_params[param_id]->count(),
        delta, this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_div(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), this->update_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->gpu_data(), Dtype(0),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
  // Add the extra history entries for AdaDelta after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
        const vector<int>& shape = net_params[i]->shape();
        this->history_.push_back(
                shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype delta = this->param_.delta();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  size_t update_history_offset = net_params.size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history of gradients
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->cpu_data(), momentum,
        this->history_[param_id]->mutable_cpu_data());

    // add delta to history to guard against dividing by zero later
    caffe_set(net_params[param_id]->count(), delta,
        this->temp_[param_id]->mutable_cpu_data());

    caffe_add(net_params[param_id]->count(),
        this->temp_[param_id]->cpu_data(),
        this->history_[update_history_offset + param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    caffe_add(net_params[param_id]->count(),
        this->temp_[param_id]->cpu_data(),
        this->history_[param_id]->cpu_data(),
        this->temp_[param_id]->mutable_cpu_data());

    // divide history of updates by history of gradients
    caffe_div(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        this->temp_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // jointly compute the RMS of both for update and gradient history
    caffe_powx(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    // compute the update
    caffe_mul(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    // compute square of update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history of updates
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->cpu_data(), momentum,
        this->history_[update_history_offset + param_id]->mutable_cpu_data());

    // apply learning rate
    caffe_cpu_scale(net_params[param_id]->count(), local_rate,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // compute square of gradient in update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history of gradients
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->gpu_data(), momentum,
        this->history_[param_id]->mutable_gpu_data());

    // add delta to history to guard against dividing by zero later
    caffe_gpu_set(net_params[param_id]->count(), delta,
        this->temp_[param_id]->mutable_gpu_data());

    caffe_gpu_add(net_params[param_id]->count(),
        this->temp_[param_id]->gpu_data(),
        this->history_[update_history_offset + param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    caffe_gpu_add(net_params[param_id]->count(),
        this->temp_[param_id]->gpu_data(),
        this->history_[param_id]->gpu_data(),
        this->temp_[param_id]->mutable_gpu_data());

    // divide history of updates by history of gradients
    caffe_gpu_div(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(),
        this->temp_[param_id]->gpu_data(),
        this->update_[param_id]->mutable_gpu_data());

    // jointly compute the RMS of both for update and gradient history
    caffe_gpu_powx(net_params[param_id]->count(),
        this->update_[param_id]->gpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_gpu_data());

    // compute the update and copy to net_diff
    caffe_gpu_mul(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(),
        this->update_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());

    // compute square of update
    caffe_gpu_powx(net_params[param_id]->count(),
        net_params[param_id]->gpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_gpu_data());

    // update history of updates
    caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
        this->update_[param_id]->gpu_data(), momentum,
        this->history_[update_history_offset + param_id]->mutable_gpu_data());

    // apply learning rate
    caffe_gpu_scale(net_params[param_id]->count(), local_rate,
        net_params[param_id]->gpu_diff(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_  + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // set update
    caffe_powx(N,
        val_v->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
    caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data());

    caffe_cpu_scale(N, local_rate*correction,
        val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_gpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->gpu_diff(), beta1,
        val_m->mutable_gpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_gpu_mul(N,
        net_params[param_id]->gpu_diff(),
        net_params[param_id]->gpu_diff(),
        val_t->mutable_gpu_data());
    caffe_gpu_axpby(N, Dtype(1)-beta2,
        val_t->gpu_data(), beta2,
        val_v->mutable_gpu_data());

    // set update
    caffe_gpu_powx(N,
        val_v->gpu_data(), Dtype(0.5),
        val_t->mutable_gpu_data());
    caffe_gpu_add_scalar(N, eps_hat,
        val_t->mutable_gpu_data());
    caffe_gpu_div(N,
        val_m->gpu_data(),
        val_t->gpu_data(),
        val_t->mutable_gpu_data());

    caffe_gpu_scale(N, local_rate*correction,
        val_t->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);
INSTANTIATE_CLASS(RMSPropSolver);
INSTANTIATE_CLASS(AdaDeltaSolver);
INSTANTIATE_CLASS(AdamSolver);

}  // namespace caffe
