/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    SetupConstants(consts);

    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
  }

  void Run(const std::vector<const DLTensor*> &inputs,
           const std::vector<const DLTensor*> &outputs) {
    // TODO: Is it thread safe to use one stream from different threads?
    //       In case of CPU engine that is safe, because stream is
    //       immediate runner and has no internal state.
    if (doPreprocess_) {
      scaledValues_.resize(inputs[2]->shape[0]);
      float* pdata1 = (float*)inputs[2]->data;
      for(size_t i = 0; i < scaledValues_.size(); ++i) {
        scaledValues_[i] = pdata1[i] * biasScale_;
      }
      doPreprocess_ = false;
    }
    dnnl::stream cur_stream = stream_;
    // map of memory object to replace with provided in/out
    std::map<dnnl_memory_t, dnnl::memory> io_replacement;

    ICHECK(inputs.size() == input_var_eid_.size());
    if (scaledValues_.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        const auto tensor = inputs[i];
        const auto eid = input_var_eid_[i];

        // Constructing a new one memory object pointed on DLTensor memory
        // TODO(@apeskov) : should be a single constructor with check of ability to wrap
        auto orig_mem = entry_out_mem_[eid].first;
        auto orig_desc = orig_mem.get_desc();
        auto new_one_mem = dnnl::memory(orig_desc, engine_, tensor->data);

        io_replacement[orig_mem.get()] = new_one_mem;
      }
    } else {
      size_t i;
      for (i = 0; i < inputs.size() - 1; i++) {
        const auto tensor = inputs[i];
        const auto eid = input_var_eid_[i];

        // Constructing a new one memory object pointed on DLTensor memory
        // TODO(@apeskov) : should be a single constructor with check of ability to wrap
        auto orig_mem = entry_out_mem_[eid].first;
        auto orig_desc = orig_mem.get_desc();
        auto new_one_mem = dnnl::memory(orig_desc, engine_, tensor->data);

        io_replacement[orig_mem.get()] = new_one_mem;
      }
      const auto tensor = inputs[i];
      const auto eid = input_var_eid_[i];
      auto orig_mem = entry_out_mem_[eid].first;
      auto orig_desc = orig_mem.get_desc();
      auto new_one_mem = dnnl::memory(orig_desc, engine_, scaledValues_.data());
      io_replacement[orig_mem.get()] = new_one_mem;

    }
    ICHECK(outputs.size() == outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      const auto tensor = outputs[i];

      auto orig_mem = entry_out_mem_[eid].first;
      auto orig_desc = orig_mem.get_desc();
      auto new_one_mem = dnnl::memory(orig_desc, engine_, tensor->data);

      io_replacement[orig_mem.get()] = new_one_mem;
    }

    auto replace_io = [&io_replacement] (std::unordered_map<int, dnnl::memory> &args) {
      for (auto &kvp : args) {
        auto orig_mem_c_hdl = kvp.second.get();
        if (io_replacement.count(orig_mem_c_hdl) != 0)
          kvp.second = io_replacement[orig_mem_c_hdl];
      }
    };

    auto replace_scratchpad = [this] (std::unordered_map<int, dnnl::memory> &args) {
      auto found = args.find(DNNL_ARG_SCRATCHPAD);
      if (found != args.end()) {
        auto scratchpad_desc = found->second.get_desc();
        auto new_scratchpad = dnnl::memory(scratchpad_desc, this->engine_);
        found->second = new_scratchpad;
      }
    };

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      auto prim = net_.at(i);
      auto args = net_args_.at(i);
      replace_io(args);
      replace_scratchpad(args);
      net_.at(i).execute(cur_stream, args);
    }
    cur_stream.wait();
  }

  /**
   * Override GetFunction() to specify Run method.
   *
   * @param name name of queried function
   * @param sptr_to_self pointer to self module
   * @return Found function in PackedFunc format, or null if no present
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
          << "Found mismatch in the number of provided data entryies and required.";

        auto copy_arg_tensors = [](TVMArgs &args, size_t start, size_t size) {
          std::vector<const DLTensor*> res(size);

          for (size_t idx = 0; idx < size; idx++) {
            const size_t arg_idx = start + idx;
            ICHECK(args[arg_idx].type_code() == kTVMNDArrayHandle ||
                   args[arg_idx].type_code() == kTVMDLTensorHandle)
              << "Expect NDArray or DLTensor as inputs";

            const DLTensor* arg;
            if (args[arg_idx].IsObjectRef<NDArray>()) {
              NDArray arr = args[arg_idx];
              arg = arr.operator->();
            } else {
              arg = args[arg_idx].operator DLTensor*();
            }
            res[idx] = arg;
          }
          return res;
        };
        auto inputs = copy_arg_tensors(args, 0, input_var_eid_.size());
        auto outputs = copy_arg_tensors(args, input_var_eid_.size(), outputs_.size());

        // Execute the subgraph.
        this->Run(inputs, outputs);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

  void Run() override {
    // Bind input buffers
    std::map<dnnl_memory_t, dnnl::memory> io_replacement;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      if (std::find(input_var_eid_.begin(), input_var_eid_.end(), eid) == input_var_eid_.end())
        continue;
      // TODO(@apeskov): check if entry_out_mem_[eid] exists
      //   check is sizes/dtype/offset is matched,
      // TODO(@comaniac): Support other data lengths.
//      auto mem = entry_out_mem_[eid].first;
//      size_t buffer_size = GetDataSize(*data_entry_[eid]);
//      mem.set_data_handle(data_entry_[eid]->data);

      // TODO(@apeskov): Thread safety. Do not touch original memory just replace it
      //  with now one.
      auto orig_mem = entry_out_mem_[eid].first;
      auto orig_desc = orig_mem.get_desc();
      auto orig_c_hdl = orig_mem.get();
      auto new_one_mem = dnnl::memory(orig_desc, engine_, data_entry_[eid]->data);
      io_replacement[orig_c_hdl] = new_one_mem;
    }
    // Bind output buffers
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
//      auto mem = entry_out_mem_[eid].first;
//      mem.set_data_handle(data_entry_[eid]->data);
      auto orig_mem = entry_out_mem_[eid].first;
      auto orig_desc = orig_mem.get_desc();
      auto orig_c_hdl = orig_mem.get();
      auto new_one_mem = dnnl::memory(orig_desc, engine_, data_entry_[eid]->data);
      io_replacement[orig_c_hdl] = new_one_mem;
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      auto prim = net_.at(i);
      auto args = net_args_.at(i);
      for (auto &kvp : args) {
        auto orig_mem_c_hdl = kvp.second.get();
        if (io_replacement.count(orig_mem_c_hdl) != 0)
          kvp.second = io_replacement[orig_mem_c_hdl];
      }
      net_.at(i).execute(stream_, args);
    }
    stream_.wait();
  }

  void ShareMemory(const DNNLJSONRuntime &other) {
    auto orig = internal_mem_;
    internal_mem_ = other.internal_mem_;
    // update args
    for (auto &args : net_args_) {
      for (auto &p : args) {
        auto mem = p.second;
         auto found = std::find(orig.begin(), orig.end(), mem);
        if (found != orig.end()) {
          auto idx = std::distance(found, orig.begin());
          p.second = internal_mem_[idx];
       }
      }
    }
  }

 private:
  bool doPreprocess_ = false; // workaround to fix bias scaling issue
  float biasScale_ = 0.0f;
  std::vector<float> scaledValues_;
  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("dnnl.qnn.conv2d" == op_name) {
          QnnConv2d(nid);
        } else if ("dnnl.qnn.conv2d_sum" == op_name) {
          QnnConv2dSum(nid);
        } else if ("dnnl.qnn.dense" == op_name) {
          QnnDense(nid);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else if ("qnn.batch_matmul" == op_name) {
          BatchMatMul(nid);
        } else if ("dnnl.qnn.batch_matmul_dequantize" == op_name) {
          BatchMatMulDequantize(nid);
        } else if ("dnnl.qnn.batch_matmul_requantize" == op_name) {
          BatchMatMulRequantize(nid);
        } else if ("qnn.dense" == op_name) {
          Denses8s8s32(nid);
        } else if ("dnnl.qnn.dense_dequantize" == op_name){
          DenseMulDequantizeGELU(nid, false);
        } else if ("dnnl.qnn.dense_dequantize_gelu" == op_name){
          DenseMulDequantizeGELU(nid);
        } else if ("dnnl.qnn.gelu" == op_name){
          GELU(nid);
        } else if ("dnnl.qnn.batchnorm" == op_name){
          DNNLBatchNorm(nid);
        } else if ("dnnl.qnn.softmax" == op_name){
          DNNLSoftMax(nid);
        }
        else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      if (data_entry_[eid])
        return BindDNNLMemoryToConst(entry, mem_desc, offset);
      else
        return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK(dltype.bits == 32 || dltype.bits == 8);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemoryToConst(const JSONGraphNodeEntry& entry, dnnl::memory::desc desc,
                                     size_t offset = 0) {
    auto eid = EntryID(entry);
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK(dltype.bits == 32 || dltype.bits == 8);

    auto dl_tensor = data_entry_[eid];
    ICHECK(dl_tensor);

    auto mem = dnnl::memory(desc, engine_, dl_tensor->data);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);


    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;
    if (has_relu) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::f32, tag::nchw});

    // Weight memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, {weights_dims, dt::f32, (groups > 1) ? tag::goihw : tag::oihw});
    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    } else {
      std::vector<float> bias(OC, 0);
      write_to_dnnl_memory(bias.data(), conv2d_bias_memory, OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  template<typename T>
  std::vector<T> get_values(const JSONGraphNodeEntry& entry, size_t size) {
    auto eid = EntryID(entry);
    auto dl_tensor = data_entry_[eid];
    ICHECK(dl_tensor->dtype.bits == sizeof(T)*8);
    ICHECK(std::is_floating_point<T>::value ? dl_tensor->dtype.code == kDLFloat : true);
    ICHECK((std::is_integral<T>::value && std::is_signed<T>::value)
               ? dl_tensor->dtype.code == kDLInt : true);
    ICHECK((std::is_integral<T>::value && std::is_unsigned<T>::value)
               ? dl_tensor->dtype.code == kDLUInt : true);
    const T* ptr = static_cast<T*>(dl_tensor->data);
    if (dl_tensor->ndim == 0) {
      // broadcast
      return std::vector<T>(size, *ptr);
    } else {
//      ICHECK(dl_tensor->shape[0] == size);
      return std::vector<T>(ptr, ptr + size);
    }
  };

template<typename T>
  T get_value(const JSONGraphNodeEntry& entry) {
    auto eid = EntryID(entry);
    auto dl_tensor = data_entry_[eid];
    ICHECK(dl_tensor->dtype.bits == sizeof(T)*8);
    const T* ptr = static_cast<T*>(dl_tensor->data);
    return *ptr;
  };


  std::vector<int32_t> quasi_conv(std::vector<int32_t>data , std::vector<int8_t> weight,
                                  int KH, int KW, int IC, int OC) {
    const auto* w_ptr = weight.data();
    std::vector<int32_t> res (OC, 0);
    for (int khw = 0; khw < KH*KW; khw++)
      for (int ic = 0; ic < IC; ic++)
        for (int oc = 0; oc < OC; oc++)
          res[oc] += data[ic] * static_cast<int32_t>(*w_ptr++);
    return res;
  }

  std::vector<int32_t> calc_bias_with_zp(const JSONGraphNode& node, int KH, int KW, int IC, int OC) {
    auto wght_entry = node.GetInputs()[1];
    auto bias_entry = node.GetInputs()[6];

    auto data_zero_point_entry = node.GetInputs()[2];
    auto input_scale_entry = node.GetInputs()[7];
    auto input_zero_point_entry = node.GetInputs()[8];
    auto output_scale_entry = node.GetInputs()[9];
    auto output_zero_point_entry = node.GetInputs()[10];

    // out_scl = rq_in_scl<f32>/rq_out_scl<f32>
    // shft_src = 0
    // shft_dst = rq_out_zp<i32> + rq_in_scl<f32>/rq_out_scl<f32> * (- zp_D - rq_in_zp<i32>)
    //   zp_D = conv(zp_A * QW)
    //
    // if we want to move into bias, then:
    // bias += rq_out_scl<f32>/rq_in_scl<f32> * rq_out_zp<i32> - zp_D - rq_in_zp<i32>

    auto zp_A = get_values<int>(data_zero_point_entry, IC);

    auto rq_out_zp = get_values<int32_t>(output_zero_point_entry, OC);
    auto rq_in_scl = get_values<float>(input_scale_entry, OC);
    auto rq_out_scl = get_values<float>(output_scale_entry, OC);
    auto rq_in_zp = get_values<int32_t>(input_zero_point_entry, OC);
    auto bias = get_values<int32_t>(bias_entry, OC);
    auto weight = get_values<int8_t>(wght_entry, KH*KW*IC*OC);

    auto zp_D = quasi_conv(zp_A, weight, KH, KW, IC, OC);

    std::vector<int32_t> res (OC, 0);

    for (int i = 0; i < OC; i++) {
      // Version for out scale
//      res[i] =  rq_out_zp[i] + static_cast<int32_t>(rq_in_scl[i]/rq_out_scl[i] * (/*bias[i]*/ - rq_in_zp[i] - zp_D[i]));
      // Version for bias update
      res[i] = bias[i] + static_cast<int32_t>(rq_out_scl[i]/rq_in_scl[i] * rq_out_zp[i]) - zp_D[i] - rq_in_zp[i];
    }
    return res;
  }

  std::vector<float> calc_out_scale(const JSONGraphNode& node, int KH, int KW, int IC, int OC) {
    auto input_scale_entry = node.GetInputs()[7];
    auto output_scale_entry = node.GetInputs()[9];

    auto rq_in_scl = get_values<float>(input_scale_entry, OC);
    auto rq_out_scl = get_values<float>(output_scale_entry, OC);

    std::vector<float> res (OC, 0);
    for (int i = 0; i < OC; i++) {
      res[i] = rq_in_scl[i]/rq_out_scl[i];
    }
    return res;
  }


/*
 * Case ConvSumRelu
 * ================
 * Original relay representation:
 *    in1<u8>    in2<u8>  w<i8>  b<i32>
 *    qnn.conv<i32>(in1, w, conv_i_zp, conv_i_zcl, conv_w_zp, conv_w_scl)
 *    add<i32>(conv, b)
 *    qnn.reqaunt<i32>(add, rq_i_zp, rq_i_scl, rq_o_zp, rq_o_scl)
 *    clip<i32>(reqaunt, 0, 255)
 *    cast<u8>(clip)
 *    qnn.add(cast, in2, lhs_scl, lhs_zp, rhs_scl, rhs_zp, output_scl, output_zp)
 *
 *    totally have 17 inputs:
 *       in1, in2, w, b
 *       conv_i_zp(==0 ??), conv_i_zcl(ignored), conv_w_zp(==0), conv_w_scl(ignored)
 *       rq_i_zp(==0 ??), rq_i_scl, rq_o_zp, rq_o_scl
 *       lhs_scl, lhs_zp, rhs_scl, rhs_zp, output_scl, output_zp
 *    Some of then SHOULD be a zero, some of them ignored
 *
 *    (conv(in1, w) + bias)*  rq_i_scl / rq_o_scl + rq_o_zp  //  and CLIP!!! can we ignore it??
 *                                                           //  or may be eltwise with clip??
 *
 *    (((conv(in1, w) + bias)*  rq_i_scl / rq_o_scl + rq_o_zp  - lhs_zp ) * lhs_scl
 *      + (in2 - rhs_zp) * rhs_scl ) / output_scl + output_zp =
 *
 *    conv(in1, w) * rq_i_scl / rq_o_scl * lhs_scl / output_scl +
 *    bias * rq_i_scl / rq_o_scl * lhs_scl / output_scl +
 *    (rq_o_zp  - lhs_zp) * lhs_scl / output_scl +
 *    (in2 - rhs_zp) * rhs_scl / output_scl +
 *    output_zp
 * DNNL suggest next function:
 *    conv2d with post op sum + out_scale + out_zp
 *
 *     dst = ( clip(alfa * (conv(i<u8>, w<i8>)) + b<i32>) + beta * in2<u8> ) + gamma<i32>
 *
 *   Have to calc alfa, beta, gamma and bias update
 *     alfa - to meet final scale
 *     beta - to align in2 scale with final scale (??? is it accurate)
 *     gama - to meet final zero point
 *     clip_low  - ??
 *     clip_high - ??
 *     originally : clip (rq_i_scl / rq_o_scl (conv(i<u8>, w<i8>) + bias) , -lhs_scl / output_scl * rq_out_zp, lhs_scl / output_scl (255-rq_out_zp))
 *
 *    alfa = rq_i_scl / rq_o_scl * lhs_scl / output_scl
 *    beta = rhs_scl / output_scl
 *    gamma = output_zp  - rhs_zp * rhs_scl / output_scl + (rq_o_zp  - lhs_zp) * lhs_scl / output_scl
 *            // or may be move to bias??
 *    clip_low = -lhs_scl / output_scl * rq_out_zp
 *    clip_high = lhs_scl / output_scl (255-rq_out_zp)
 */
std::tuple<
      std::vector<float>,
      std::vector<float>,
      std::vector<int32_t>,
      float,
      float
  > calc_abg_for_sum(const JSONGraphNode& node, int OC) {
    std::vector<float> alfa(OC), beta(OC);
    std::vector<int32_t> gamma(OC);
    // 0-1 in1 weight
    // 2-5 conv
    // 6 - bias
    // 7-10 - requant args
    // 11 - in2
    // 12-17 - qnn add args

    auto in1_entry = node.GetInputs()[0];   // not used
    auto wght_entry = node.GetInputs()[1];  // not used
    // conv quant args
    auto data_zp_entry = node.GetInputs()[2];  // should be zero
    auto wght_zp_entry = node.GetInputs()[3];  // should be zero
    auto data_scl_entry = node.GetInputs()[4]; // not used
    auto wght_scl_entry = node.GetInputs()[5]; // not used
    // bias
    auto bias_entry = node.GetInputs()[6];
    // requant args
    auto rq_i_scl_entry = node.GetInputs()[7];
    auto rq_i_zp_entry = node.GetInputs()[8];  // should be zero
    auto rq_o_scl_entry = node.GetInputs()[9];
    auto rq_o_zp_entry = node.GetInputs()[10];
    // sum_in2
    auto sum_in2_entry = node.GetInputs()[11];
    // qnn add quant args
    auto lhs_scl_entry = node.GetInputs()[12];
    auto lhs_zp_entry = node.GetInputs()[13];
    auto rhs_scl_entry = node.GetInputs()[14];
    auto rhs_zp_entry = node.GetInputs()[15];
    auto out_scl_entry = node.GetInputs()[16];
    auto out_zp_entry = node.GetInputs()[17];


    auto rq_i_scl = get_values<float>(rq_i_scl_entry, OC);
    auto rq_o_scl = get_values<float>(rq_o_scl_entry, OC);
    auto lhs_scl = get_values<float>(lhs_scl_entry, OC);
    auto rhs_scl = get_values<float>(rhs_scl_entry, OC);
    auto out_scl = get_values<float>(out_scl_entry, OC);

    auto rq_o_zp = get_values<int32_t>(rq_o_zp_entry, OC);
    auto out_zp = get_values<int32_t>(out_zp_entry, OC);
    auto rhs_zp = get_values<int32_t>(rhs_zp_entry, OC);
    auto lhs_zp = get_values<int32_t>(lhs_zp_entry, OC);

    // alfa = rq_i_scl / rq_o_scl * lhs_scl / out_scl
    // beta = rhs_scl / output_scl
    // gamma = out_zp  - rhs_zp * rhs_scl / out_scl + (rq_o_zp  - lhs_zp) * lhs_scl / out_scl

    for (int i = 0; i < OC; i++) {
      alfa[i] = rq_i_scl[i] / rq_o_scl[i] * lhs_scl[i] / out_scl[i];
      beta[i] = rhs_scl[i] / out_scl[i];
      gamma[i] = out_zp[i]  - rhs_zp[i] * rhs_scl[i] / out_scl[i] + (rq_o_zp[i]  - lhs_zp[i])
                                                                  * lhs_scl[i] / out_scl[i];
    }

    float clip_low = - lhs_scl[0] / out_scl[0] * rq_o_zp[0];
    float clip_high = lhs_scl[0] / out_scl[0] * (255 - rq_o_zp[0]);

    return {alfa, beta, gamma, clip_low, clip_high};
  }

  /**
   *
   *
   *
   * Case ConvSumRelu
   * ================
   * Original relay representation:
   *    in1<u8>    in2<u8>  w<i8>  b<i32>
   *    qnn.conv<i32>(in1, w, conv_i_zp, conv_i_zcl, conv_w_zp, conv_w_scl)
   *    add<i32>(conv, b)
   *    qnn.reqaunt<i32>(add, rq_i_zp, rq_i_scl, rq_o_zp, rq_o_scl)
   *    clip<i32>(reqaunt, 0, 255)
   *    cast<u8>(clip)
   *    qnn.add(cast, in2, lhs_scl, lhs_zp, rhs_scl, rhs_zp, output_scl, output_zp)
   *
   *    totally have 17 inputs:
   *       in1, in2, w, b
   *       conv_i_zp(==0 ??), conv_i_zcl(ignored), conv_w_zp(==0), conv_w_scl(ignored)
   *       rq_i_zp(==0 ??), rq_i_scl, rq_o_zp, rq_o_scl
   *       lhs_scl, lhs_zp, rhs_scl, rhs_zp, output_scl, output_zp
   *    Some of then SHOULD be a zero, some of them ignored
   *
   *    (conv(in1, w) + bias)*  rq_i_scl / rq_o_scl + rq_o_zp  //  and CLIP!!! can we ignore it??
   *                                                           //  or may be eltwise with clip??
   *
   *    (((conv(in1, w) + bias)*  rq_i_scl / rq_o_scl + rq_o_zp  - lhs_zp ) * lhs_scl
   *      + (in2 - rhs_zp) * rhs_scl ) / output_scl + output_zp =
   *
   *    conv(in1, w) * rq_i_scl / rq_o_scl * lhs_scl / output_scl +
   *    bias * rq_i_scl / rq_o_scl * lhs_scl / output_scl +
   *    (rq_o_zp  - lhs_zp) * lhs_scl / output_scl +
   *    (in2 - rhs_zp) * rhs_scl / output_scl +
   *    output_zp
   *
   *
   *
   *
   *
   * DNNL suggest next function:
   *    conv2d with post op sum + out_scale + out_zp
   *
   *     dst = ( alfa * (conv(i<u8>, w<i8>) + b<i32>) + beta * in2<u8> ) + gamma<i32>
   *
   *   Have to calc alfa, beta, gamma and bias update
   *     alfa - to meet final scale
   *     beta - to align in2 scale with final scale (??? is it accurate)
   *     gama - to meet final zero point
   *
   *    alfa = rq_i_scl / rq_o_scl * lhs_scl / output_scl
   *    beta = rhs_scl / output_scl
   *    gamma = output_zp  - rhs_zp * rhs_scl / output_scl + (rq_o_zp  - lhs_zp) * lhs_scl / output_scl
   *            // or may be move to bias??
   *
   *
   * @param nid
   */
  void QnnConv2d(const size_t& nid, bool with_sum = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
        IH = input_shape[1],                    // input channels
        IW = input_shape[2],                    // input height
        IC = input_shape[3],                    // input width
        KH = weight_shape[0],                   // output channels
        KW = weight_shape[1],                   // weight height
        OC = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::u8, tag::nhwc);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::s8, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::s32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::u8, tag::nhwc);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;

    // out rescale
    auto out_scale = calc_out_scale(node, KH, KW, IC, OC);
    attr.set_output_scales(1 << 1, out_scale);

    // out zero point
    // NB! There is a limitation of DNNL. 1<<1 mask is not supported for
    //     zero point specification. Have to inject it into bias values
    auto out_zero_point = calc_bias_with_zp(node, KH, KW, IC, OC);
    dnnl::memory conv2d_bias_memory({{OC}, dt::s32, tag::a}, engine_);
    std::copy(out_zero_point.begin(), out_zero_point.end(),
              static_cast<int32_t*>(conv2d_bias_memory.get_data_handle()));

    // Explicit scratchpad specification
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NHWC");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::u8, tag::nhwc});

    // Weight memory is in constants.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "HWIO");
    auto weights_memory =
        BindDNNLMemory(weight_entry, {weights_dims, dt::s8, (groups > 1) ? tag::hwigo : tag::hwio});

    auto suggested_wgh_desc = conv2d_prim_desc.weights_desc();
    dnnl::memory conv2d_weights_memory = dnnl::memory(suggested_wgh_desc, engine_);
    auto w_reorder = dnnl::reorder(weights_memory, conv2d_weights_memory);
    w_reorder.execute(stream_, weights_memory, conv2d_weights_memory);

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_.push_back(conv);
    std::unordered_map<int, dnnl::memory> args {{DNNL_ARG_SRC, conv2d_src_memory},
                                                {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                                                {DNNL_ARG_BIAS, conv2d_bias_memory},
                                                {DNNL_ARG_DST, conv2d_dst_memory}};

    auto scratchpad_desc = conv2d_prim_desc.scratchpad_desc();
    if (scratchpad_desc) {
      dnnl::memory scratchpad_mem(scratchpad_desc, engine_);
      internal_mem_.push_back(scratchpad_mem);
      args[DNNL_ARG_SCRATCHPAD] = scratchpad_mem;
    }

    net_args_.push_back(args);

    internal_mem_.push_back(conv2d_weights_memory);
    internal_mem_.push_back(conv2d_bias_memory);
  }

  void QnnConv2dSum(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto bias_entry = node.GetInputs()[6];
    auto sum_input_entry = node.GetInputs()[11];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
    IH = input_shape[1],                    // input channels
    IW = input_shape[2],                    // input height
    IC = input_shape[3],                    // input width
    KH = weight_shape[0],                   // output channels
    KW = weight_shape[1],                   // weight height
    OC = weight_shape[3],                   // weight width
    PH_L = std::stoi(str_padding[1]),       // height padding: left
    PH_R = std::stoi(str_padding[3]),       // height padding: right
    PW_L = std::stoi(str_padding[0]),       // width padding: left
    PW_R = std::stoi(str_padding[2]),       // width padding: right
    SH = std::stoi(str_strides[0]),         // height-wise stride
    SW = std::stoi(str_strides[0]),         // weight-wise stride
    OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
    OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::u8, tag::nhwc);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::s8, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::s32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::u8, tag::nhwc);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;


    std::vector<float> alfa, beta;
    std::vector<int32_t> gamma;
    float clip_low, clip_high;

    std::tie(alfa, beta, gamma, clip_low, clip_high) = calc_abg_for_sum(node, OC);
    // out rescale
    attr.set_output_scales(1 << 1, alfa);
    attr.set_zero_points(DNNL_ARG_DST, 0, {gamma[0]}); // TODO: check if it's broadcasted scalar
    dnnl::post_ops pops;
    pops.append_eltwise(1.0, dnnl::algorithm::eltwise_clip, clip_low, clip_high);
    pops.append_sum(beta[0], dnnl::memory::data_type::u8);

    attr.set_post_ops(pops);

    // Explicit scratchpad specification
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NHWC");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::u8, tag::nhwc});

    // Weight memory is in constants.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "HWIO");
    auto weights_memory =
        BindDNNLMemory(weight_entry, {weights_dims, dt::s8, (groups > 1) ? tag::hwigo : tag::hwio});

    auto suggested_wgh_desc = conv2d_prim_desc.weights_desc();
    dnnl::memory conv2d_weights_memory = dnnl::memory(suggested_wgh_desc, engine_);
    auto w_reorder = dnnl::reorder(weights_memory, conv2d_weights_memory);
    w_reorder.execute(stream_, weights_memory, conv2d_weights_memory);

    // Weight memory is in constants.
    auto conv2d_bias_memory =
        BindDNNLMemory(bias_entry, {{OC}, dt::s32, tag::a});

    auto conv2d_sum_input_memory =
        BindDNNLMemory(sum_input_entry, {dst_dims, dt::u8, tag::nhwc});
    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // TODO: fake copy right here
//    {
//      net_.push_back(dnnl::reorder(conv2d_sum_input_memory, conv2d_dst_memory));
//      net_args_.push_back({{DNNL_ARG_SRC, conv2d_sum_input_memory},
//                           {DNNL_ARG_DST, conv2d_dst_memory}});
//    }

    // Bind memory buffers.
    net_.push_back(conv);
    std::unordered_map<int, dnnl::memory> args {
      {DNNL_ARG_SRC, conv2d_src_memory},
      {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
      {DNNL_ARG_BIAS, conv2d_bias_memory},
      {DNNL_ARG_DST, conv2d_dst_memory}
    };

    auto scratchpad_desc = conv2d_prim_desc.scratchpad_desc();
    if (scratchpad_desc) {
      dnnl::memory scratchpad_mem(scratchpad_desc, engine_);
      internal_mem_.push_back(scratchpad_mem);
      args[DNNL_ARG_SCRATCHPAD] = scratchpad_mem;
    }

    net_args_.push_back(args);
    internal_mem_.push_back(conv2d_weights_memory);
    internal_mem_.push_back(conv2d_bias_memory);
  }

  static std::vector<int32_t> quasi_dense(std::vector<int32_t>data , std::vector<int8_t> weight,
                                          int IC, int OC) {
    const auto* w_ptr = weight.data();
    std::vector<int32_t> res (OC, 0);
    for (int oc = 0; oc < OC; oc++)
      for (int ic = 0; ic < IC; ic++)
          res[oc] += data[ic] * static_cast<int32_t>(*w_ptr++);
    return res;
  }

  std::tuple<
      std::vector<float>,   // o_scl
      std::vector<int32_t>, // new_bias
      float,  // clip_low
      float   // clip_high
  > calc_qnn_dense_args(const JSONGraphNode& node, int IC, int OC) {
    std::vector<float> o_scl(OC);
    std::vector<int32_t> new_bias(OC);
    float clip_low = 0, clip_high = 0;

    // 0-1 in weight
    // 2-5 demse
    // 6 - bias
    // 7-10 - requant args
    auto in1_entry = node.GetInputs()[0];     // not used
    auto weight_entry = node.GetInputs()[1];  // not used
    // conv quant args
    auto data_zp_entry = node.GetInputs()[2];  // should be zero
    auto wght_zp_entry = node.GetInputs()[3];  // should be zero
    auto data_scl_entry = node.GetInputs()[4]; // not used
    auto wght_scl_entry = node.GetInputs()[5]; // not used
    // bias
    auto bias_entry = node.GetInputs()[6];
    // requant args
    auto rq_i_scl_entry = node.GetInputs()[7];
    auto rq_i_zp_entry = node.GetInputs()[8];  // should be zero
    auto rq_o_scl_entry = node.GetInputs()[9];
    auto rq_o_zp_entry = node.GetInputs()[10];

    auto data_zp = get_values<int32_t>(data_zp_entry, IC);
    auto weight = get_values<int8_t>(weight_entry, OC*IC);
    auto bias = get_values<int32_t>(bias_entry, OC);
    auto rq_i_scl = get_values<float>(rq_i_scl_entry, OC);
    auto rq_i_zp = get_values<int32_t>(rq_i_zp_entry, OC);
    auto rq_o_scl = get_values<float>(rq_o_scl_entry, OC);
    auto rq_o_zp = get_values<int32_t>(rq_o_zp_entry, OC);

    auto zp_D = quasi_dense(data_zp, weight, IC, OC);

    for (int i = 0; i < OC; i++) {
      o_scl[i] = rq_i_scl[i] / rq_o_scl[i];
      new_bias[i] = bias[i] + static_cast<int32_t>(rq_o_scl[i]/rq_i_scl[i] * rq_o_zp[i]) - zp_D[i] - rq_i_zp[i];
    }

    return {o_scl, new_bias, clip_low, clip_high};
  }

  void QnnDense(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto bias_entry = node.GetInputs()[6];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim
        NB = input_shape[0],   // batch size
        IC = input_shape[1],   // input channels
        OC = weight_shape[0];  // weight width

    dnnl::memory::dims data_dims = {NB, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {NB, OC};


    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::u8, tag::ab});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::ab});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::s32, tag::a});
    auto dst_md = dnnl::memory::desc({out_dims, dt::u8, tag::ab});

    // specify output scale and bias update
    std::vector<float> o_scale (OC, 0);
    std::vector<int32_t> new_bias (OC, 0);
    float clip_low, clip_high;

    std::tie(o_scale, new_bias, clip_low, clip_high) =
        calc_qnn_dense_args(node, IC, OC);

    auto bias_memory = dnnl::memory(bias_md, engine_);
    std::copy(new_bias.begin(), new_bias.end(),
              static_cast<int32_t*>(bias_memory.get_data_handle()));

    dnnl::primitive_attr attr;
    attr.set_output_scales(1<<1, o_scale);

    // Explicit scratchpad specification
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_prim_desc);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);

    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_.push_back(dense);
    std::unordered_map<int, dnnl::memory> args {
      {DNNL_ARG_SRC, data_memory},
      {DNNL_ARG_WEIGHTS, weight_memory},
      {DNNL_ARG_BIAS, bias_memory},
      {DNNL_ARG_DST, dst_memory}
    };

    auto scratchpad_desc = dense_prim_desc.scratchpad_desc();
    if (scratchpad_desc) {
      dnnl::memory scratchpad_mem(scratchpad_desc, engine_);
      internal_mem_.push_back(scratchpad_mem);
      args[DNNL_ARG_SCRATCHPAD] = scratchpad_mem;
    }

    net_args_.push_back(args);

    internal_mem_.push_back(weight_memory);
    internal_mem_.push_back(bias_memory);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
    IC = input_shape[1],               // input channels
    OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    std::vector<float> bias(OC, 0);
    write_to_dnnl_memory(bias.data(), bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());


    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void Denses8s8s32(const size_t& nid) {
    auto node = nodes_[nid];
    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
    IC = input_shape[1],               // input channels
    OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    // dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::s8, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::nc});
    // auto bias_md = dnnl::memory::desc({bias_dims, dt::s32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::s32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());


    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                        //  {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_md = dnnl::memory::desc{{shape}, dt::f32, tag::abcd};

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto out_md = dnnl::memory::desc(shape, dt::f32, tag::abcd);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto add_desc =
        dnnl::binary::desc(dnnl::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = dnnl::binary::primitive_desc(add_desc, engine_);
    auto add = dnnl::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  void BatchMatMul(const size_t& nid) {
    // Setup attributes.
    auto node = nodes_[nid];
    ICHECK_EQ(node.GetInputs().size(), 6);
    auto out_shape = node.GetOpShape();
    ICHECK_EQ(out_shape.size(), 1);
    ICHECK_EQ(out_shape[0].size(), 3);
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    // sshtin ToDo: use these values in case of dequantization
    // auto src_zero_point = node.GetInputs()[2];
    // auto dst_zero_point = node.GetInputs()[3];
    // auto src_scale = node.GetInputs()[4];
    // auto dst_scale = node.GetInputs()[5];

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    ICHECK_EQ(input_shape.size(), 3) << "input shape should have 3 dimentions.";
    ICHECK_EQ(weight_shape.size(), 3) << "weights shape should have 3 dimentions.";
    ICHECK_EQ(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_].bits, 8);
    ICHECK_EQ(nodes_[weight_entry.id_].GetOpDataType()[weight_entry.index_].bits, 8);
    dnnl::memory::dims data_dims   = {input_shape[0], input_shape[1], input_shape[2]};
    dnnl::memory::dims weight_dims = {weight_shape[0], weight_shape[2], weight_shape[1]};
    dnnl::memory::dims out_dims    = {out_shape[0][0], out_shape[0][1], out_shape[0][2]};
    auto data_md   = dnnl::memory::desc({data_dims, dt::s8, tag::abc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::acb});
    auto dst_md    = dnnl::memory::desc({out_dims, dt::s32,  tag::abc});

    // Matmul description.
    auto matmul_desc      = dnnl::matmul::desc(data_md, weight_md, dst_md);
    auto matmul_prim_desc = dnnl::matmul::primitive_desc(matmul_desc, engine_);
    auto matmul = dnnl::matmul(matmul_prim_desc);

    net_.push_back(matmul);
    // Memory mappings.
    auto data_memory   = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dst_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchMatMulDequantize(const size_t& nid) {
    // Setup attributes.
    auto node = nodes_[nid];
    ICHECK_EQ(node.GetInputs().size(), 8);
    std::cout << symbol_name_ << std::endl;
    std::cout << node.GetOpName() << std::endl;
    for (auto x : node.GetInputs()) {
      auto shape  = nodes_[x.id_].GetOpShape()[x.index_];
      std::cout << nodes_[x.id_].GetOpDataType()[x.index_] << " ";
      for (auto y : shape) {
        std::cout << y << " ";
      }
      std::cout << "\n";
    }

    auto out_shape = node.GetOpShape();
    ICHECK_EQ(out_shape.size(), 1);
    ICHECK((out_shape[0].size() == 3) || (out_shape[0].size() == 4 && out_shape[0][0] == 1));
    auto data_entry     = node.GetInputs()[0];
    auto weight_entry   = node.GetInputs()[1];
    auto src_zero_point = node.GetInputs()[2];
    auto dst_zero_point = node.GetInputs()[3];
    auto src_scale      = node.GetInputs()[4];
    auto dst_scale      = node.GetInputs()[5];

    float src_scale_val = get_value<float>(src_scale);
    float dst_scale_val = get_value<float>(dst_scale);
    int32_t src_zero_point_val = get_value<int32_t>(src_zero_point);
    int32_t dst_zero_point_val = get_value<int32_t>(dst_zero_point);

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    ICHECK_EQ(input_shape.size(), 3) << "input shape should have 3 dimentions.";
    ICHECK_EQ(weight_shape.size(), 3) << "weights shape should have 3 dimentions.";
    ICHECK_EQ(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_].bits, 8);
    ICHECK_EQ(nodes_[weight_entry.id_].GetOpDataType()[weight_entry.index_].bits, 8);
    dnnl::memory::dims data_dims   = {input_shape[0], input_shape[1], input_shape[2]};
    dnnl::memory::dims weight_dims = {weight_shape[0], weight_shape[2], weight_shape[1]};
    dnnl::memory::dims out_dims;
    if (out_shape[0].size() == 4) {
      out_dims = {out_shape[0][1], out_shape[0][2], out_shape[0][3]};
    } else {
      out_dims = {out_shape[0][0], out_shape[0][1], out_shape[0][2]};
    }
    auto data_md   = dnnl::memory::desc({data_dims, dt::s8, tag::abc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::acb});
    auto dst_md    = dnnl::memory::desc({out_dims, dt::f32, tag::abc});

    // Matmul description.
    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {src_scale_val * dst_scale_val});
    attr.set_zero_points(DNNL_ARG_DST, 0, {dst_zero_point_val});

    auto matmul_desc      = dnnl::matmul::desc(data_md, weight_md, dst_md);
    auto matmul_prim_desc = dnnl::matmul::primitive_desc(matmul_desc, attr, engine_);
    auto matmul = dnnl::matmul(matmul_prim_desc);

    net_.push_back(matmul);
    // Memory mappings.
    auto data_memory   = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dst_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchMatMulRequantize(const size_t& nid) {
    // Setup attributes.
    auto node = nodes_[nid];
    std::cout << symbol_name_ << std::endl;
    std::cout << node.GetOpName() << std::endl;
    for (auto x : node.GetInputs()) {
      auto shape  = nodes_[x.id_].GetOpShape()[x.index_];
      std::cout << nodes_[x.id_].GetOpDataType()[x.index_] << " ";
      for (auto y : shape) {
        std::cout << y << " ";
      }
      std::cout << "\n";
    }
    ICHECK_EQ(node.GetInputs().size(), 10);
    auto out_shape = node.GetOpShape();
    ICHECK_EQ(out_shape.size(), 1);
    ICHECK((out_shape[0].size() == 3) || (out_shape[0].size() == 4 && out_shape[0][0] == 1));
    auto data_entry     = node.GetInputs()[0];
    auto weight_entry   = node.GetInputs()[1];
    auto src_zero_point = node.GetInputs()[2];
    auto dst_zero_point = node.GetInputs()[3];
    auto src_scale      = node.GetInputs()[4];
    auto dst_scale      = node.GetInputs()[5];
    auto rescale_src_scale      = node.GetInputs()[6];
    auto rescale_src_zero_point = node.GetInputs()[7];
    auto rescale_dst_scale      = node.GetInputs()[8];
    auto rescale_dst_zero_point = node.GetInputs()[9];

    float src_scale_val = get_value<float>(src_scale);
    float dst_scale_val = get_value<float>(dst_scale);
    int32_t src_zero_point_val = get_value<int32_t>(src_zero_point);
    int32_t dst_zero_point_val = get_value<int32_t>(dst_zero_point);

    float rescale_src_scale_val = get_value<float>(rescale_src_scale);
    float rescale_dst_scale_val = get_value<float>(rescale_dst_scale);
    int32_t rescale_src_zero_point_val = get_value<int32_t>(rescale_src_zero_point);
    int32_t rescale_dst_zero_point_val = get_value<int32_t>(rescale_dst_zero_point);

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    ICHECK_EQ(input_shape.size(), 3) << "input shape should have 3 dimentions.";
    ICHECK_EQ(weight_shape.size(), 3) << "weights shape should have 3 dimentions.";
    ICHECK_EQ(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_].bits, 8);
    ICHECK_EQ(nodes_[weight_entry.id_].GetOpDataType()[weight_entry.index_].bits, 8);
    dnnl::memory::dims data_dims   = {input_shape[0],  input_shape[1],  input_shape[2]};
    dnnl::memory::dims weight_dims = {weight_shape[0], weight_shape[2], weight_shape[1]};
    dnnl::memory::dims out_dims    = {weight_shape[0], input_shape[1],  weight_shape[1]};
    auto data_md   = dnnl::memory::desc({data_dims, dt::s8, tag::abc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::acb});
    auto interm_md = dnnl::memory::desc({out_dims, dt::s8, tag::abc});
    auto dst_md    = dnnl::memory::desc({out_shape[0], dt::s8, tag::abc});
    // Matmul description.
    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {src_scale_val* dst_scale_val});
    // attr.set_zero_points(DNNL_ARG_SRC, 0, {src_zero_point_val});
    attr.set_zero_points(DNNL_ARG_DST, 0, {dst_zero_point_val});
    
    auto matmul_desc      = dnnl::matmul::desc(data_md, weight_md, interm_md);
    auto matmul_prim_desc = dnnl::matmul::primitive_desc(matmul_desc, attr, engine_);
    auto matmul = dnnl::matmul(matmul_prim_desc);

    net_.push_back(matmul);
    // Memory mappings.
    auto data_memory   = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto interm_memory = dnnl::memory(interm_md, engine_);


    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dst_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                          {DNNL_ARG_WEIGHTS, weight_memory},
                          {DNNL_ARG_DST, interm_memory}});
    // dnnl::primitive_attr reorder_attr;
    // reorder_attr.set_output_scales(0, {rescale_src_scale_val * rescale_dst_scale_val});
    auto toTranspose_pd = dnnl::memory::desc({1, out_dims[0], out_dims[1], out_dims[2]}, dt::f32, tag::nchw, interm_memory.get_data_handle());
    auto toTranspose_memory = dnnl::memory(toTranspose_pd, engine_);
    auto dstTmp_md    = dnnl::memory::desc({1, out_dims[0], out_dims[1], out_dims[2]}, dt::s8, tag::nhwc, dst_memory.get_data_handle());
    auto reorder_pd = dnnl::reorder::primitive_desc(
            engine_, toTranspose_pd, engine_, dstTmp_md);//, reorder_attr);
    auto reorder_prim = dnnl::reorder(reorder_pd);
    net_.push_back(reorder_prim);
    net_args_.push_back({{DNNL_ARG_SRC, toTranspose_memory},
                         {DNNL_ARG_DST, dst_memory}});

    internal_mem_.push_back(interm_memory);
  }


  void DenseMulDequantizeGELU(const size_t& nid, bool postop = true) {
    // Setup attributes.
    auto node = nodes_[nid];
    ICHECK_EQ(node.GetInputs().size(), 9);
    auto out_shape = node.GetOpShape();
    ICHECK_EQ(out_shape.size(), 1);
    ICHECK((out_shape[0].size() == 2) || ((out_shape[0].size() == 3 && out_shape[0][0] == 1)));
    auto data_entry     = node.GetInputs()[0];
    auto weight_entry   = node.GetInputs()[1];
    auto src_zero_point = node.GetInputs()[2];
    auto dst_zero_point = node.GetInputs()[3];
    auto src_scale      = node.GetInputs()[4];
    auto dst_scale      = node.GetInputs()[5];
    auto bias_entry     = node.GetInputs()[8];
    float   src_scale_val      = get_value<float>(src_scale);
    int32_t src_zero_point_val = get_value<int32_t>(src_zero_point);

    float   dst_scale_val      = get_value<float>(dst_scale);
    int32_t dst_zero_point_val = get_value<int32_t>(dst_zero_point);

    dnnl::memory::dims input_shape  = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims bias_shape   = nodes_[bias_entry.id_].GetOpShape()[bias_entry.index_];

    ICHECK_EQ(input_shape.size(), 2) << "input shape should have 2 dimentions.";
    ICHECK_EQ(weight_shape.size(), 2) << "weights shape should have 2 dimentions.";
    ICHECK_EQ(nodes_[data_entry.id_].GetOpDataType()[data_entry.index_].bits, 8);
    ICHECK_EQ(nodes_[weight_entry.id_].GetOpDataType()[weight_entry.index_].bits, 8);

    dnnl::memory::dims data_dims   = {input_shape[0], input_shape[1]};
    dnnl::memory::dims weight_dims = {weight_shape[0], input_shape[1]};
    dnnl::memory::dims bias_dims   = {bias_shape[0]};
    dnnl::memory::dims out_dims    = {input_shape[0], weight_shape[0]};

    auto data_md   = dnnl::memory::desc({data_dims, dt::s8, tag::ab});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::s8, tag::ab});
    auto bias_md   = dnnl::memory::desc({bias_dims, dt::f32, tag::a});
    auto dst_md    = dnnl::memory::desc({out_dims, dt::f32, tag::ab});
    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {src_scale_val * dst_scale_val});
    attr.set_zero_points(DNNL_ARG_SRC, 0, {src_zero_point_val});
    attr.set_zero_points(DNNL_ARG_DST, 0, {dst_zero_point_val});
    if (postop) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_gelu_erf, 1.f, 0.f);
      attr.set_post_ops(ops);
    }
    // Explicit scratchpad specification
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_prim_desc);

    net_.push_back(dense);

    // // Memory mappings.
    auto data_memory   = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dst_md);

    biasScale_ = 1.0f / (src_scale_val * dst_scale_val);
    // sshtin: workaround
    auto eid = EntryID(bias_entry);
    auto dl_tensor = data_entry_[eid];
    dnnl::memory bias_memory;
    if (dl_tensor != nullptr) {
      dnnl::memory bias_memory(bias_md, engine_);
      float* pDst = static_cast<float*>(bias_memory.get_data_handle());
      const float* pSrc = static_cast<float*>(dl_tensor->data);
      for (int i = 0; i < bias_shape[0]; ++i) {
        pDst[i] = (pSrc[i] * biasScale_);
      }

      net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                          {DNNL_ARG_WEIGHTS, weight_memory},
                          {DNNL_ARG_BIAS, bias_memory},
                          {DNNL_ARG_DST, dst_memory}});

    } else {
      doPreprocess_ = true;
      auto bias_memory   = BindDNNLMemory(bias_entry, bias_md);
      net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                          {DNNL_ARG_WEIGHTS, weight_memory},
                          {DNNL_ARG_BIAS, bias_memory},
                          {DNNL_ARG_DST, dst_memory}});

    }
  }

  void GELU(const size_t& nid) {
    // Setup attributes.
    const std::unordered_map<size_t, dnnl::memory::format_tag> shape2Tag = {
      {2, tag::ab},
      {3, tag::abc},
      {4, tag::abcd},
      {5, tag::abcde},
    };
    auto node = nodes_[nid];

    ICHECK_EQ(node.GetInputs().size(), 1);

    auto out_shape = node.GetOpShape();
    ICHECK_EQ(out_shape.size(), 1);
    auto data_entry     = node.GetInputs()[0];

    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];

    dnnl::memory::dims data_dims = input_shape;
    dnnl::memory::dims out_dims  = out_shape[0];
    auto it = shape2Tag.find(data_dims.size());
    ICHECK(it != shape2Tag.end()) << "input shape should has invalid dimention.";
    auto data_md   = dnnl::memory::desc({data_dims, dt::f32, it->second});
    it = shape2Tag.find(data_dims.size());
    ICHECK(it != shape2Tag.end()) << "output shape should has invalid dimention.";
    auto dst_md    = dnnl::memory::desc({out_dims, dt::f32, it->second});

    auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training,
                dnnl::algorithm::eltwise_gelu_erf, data_md);
    auto eltwise_prim_desc = dnnl::eltwise_forward::primitive_desc(eltwise_desc, engine_);

    auto gelu = dnnl::eltwise_forward(eltwise_prim_desc);
    net_.push_back(gelu);

    // // // Memory mappings.
    auto data_memory   = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dst_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                          {DNNL_ARG_DST, dst_memory}});
  }

  void DNNLBatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    std::cout << node.GetInputs().size() << std::endl;
    std::cout << node.GetOpName() << std::endl;
    for (auto x : node.GetInputs()) {
      auto shape  = nodes_[x.id_].GetOpShape()[x.index_];
      std::cout << nodes_[x.id_].GetOpDataType()[x.index_] << " ";
      for (auto y : shape) {
        std::cout << y << " ";
      }
      std::cout << "\n";
    }

    ICHECK_EQ(node.GetInputs().size(), 4);
    auto input_entry = node.GetInputs()[0];
    auto delta_entry = node.GetInputs()[1];
    auto gamma_entry = node.GetInputs()[2];
    auto beta_entry  = node.GetInputs()[3];
    float delta_val  = get_value<float>(delta_entry);
    auto out_shape   = node.GetOpShape();
    dnnl::memory::dims input_shape = nodes_[input_entry.id_].GetOpShape()[input_entry.index_];
    dnnl::memory::dims gamma_shape = nodes_[gamma_entry.id_].GetOpShape()[gamma_entry.index_];
    dnnl::memory::dims beta_shape  = nodes_[beta_entry.id_].GetOpShape()[beta_entry.index_];

    dnnl::memory::dims out_dims  = out_shape[0];
    auto data_md  = GenDNNLMemDescByShape({1, input_shape[0], input_shape[1], input_shape[2]}, dt::f32);
    auto out_md   = GenDNNLMemDescByShape({1, input_shape[0], input_shape[1], input_shape[2]}, dt::f32);
    auto gamma_md = GenDNNLMemDescByShape({2, gamma_shape[0]}, dt::f32);
    auto beta_md  = GenDNNLMemDescByShape({2, beta_shape[0]},  dt::f32);
    auto batchnorm_desc = dnnl::batch_normalization_forward::desc(dnnl::prop_kind::forward_training,
                                          data_md, delta_val,
                                          dnnl::normalization_flags::use_scale_shift);

    auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(batchnorm_desc, engine_);
    auto bnorm = dnnl::batch_normalization_forward(bnorm_pd);
    net_.push_back(bnorm);
    // Memory mappings.
    auto data_memory  = BindDNNLMemory(input_entry, data_md);
    auto gamma_memory = BindDNNLMemory(gamma_entry, gamma_md);
    auto beta_memory  = BindDNNLMemory(beta_entry,  beta_md);
    auto mean_mem = dnnl::memory(bnorm_pd.mean_desc(), engine_);
    auto variance_mem = dnnl::memory(bnorm_pd.variance_desc(), engine_);
    auto workspace_mem = dnnl::memory(bnorm_pd.workspace_desc(), engine_);
    auto scale_vals = get_values<float>(gamma_entry, gamma_shape[0]);
    auto shift_vals = get_values<float>(beta_entry,  beta_shape[0]);
    float* pDst = static_cast<float*>(beta_memory.get_data_handle());
    for (int i = 0; i < beta_shape[0]; ++i) {
      pDst[i] = scale_vals[i];
      pDst[i + beta_shape[0]] = shift_vals[i];
    }
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                          {DNNL_ARG_SCALE_SHIFT, beta_memory},
                          {DNNL_ARG_VARIANCE, variance_mem},
                          {DNNL_ARG_MEAN, mean_mem},
                          {DNNL_ARG_WORKSPACE, workspace_mem},
                          {DNNL_ARG_DST, dst_memory}});

  }

  void DNNLSoftMax(const size_t& nid) {
    auto node = nodes_[nid];

    ICHECK_EQ(node.GetInputs().size(), 1);
    auto input_entry = node.GetInputs()[0];
    auto out_shape   = node.GetOpShape();
    dnnl::memory::dims input_shape = nodes_[input_entry.id_].GetOpShape()[input_entry.index_];

    auto data_md  = GenDNNLMemDescByShape(input_shape, dt::f32);
    auto out_md   = GenDNNLMemDescByShape(out_shape[0], dt::f32);
    auto softmax_d
            = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, data_md, 3); //sshtin: need to fix it

    // Create primitive descriptor.
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, engine_);

    // Create the primitive.
    auto softmax = dnnl::softmax_forward(softmax_pd);
    net_.push_back(softmax);
    // // Memory mappings.
    auto data_memory  = BindDNNLMemory(input_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                           {DNNL_ARG_DST, dst_memory}});

  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    ICHECK(mem);
    ICHECK(mem.get_desc().get_size() == size)  << "Wrong mem size";
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    ICHECK(mem);
    ICHECK(mem.get_desc().get_size() == size)  << "Wrong mem size";
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }
  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
  /* Internal tensors. Are not matched with entry ID */
  std::vector<dnnl::memory> internal_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
