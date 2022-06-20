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
 * \file src/relay/backend/contrib/dnnl/comp_op_matcher.h
 * \brief Implement matcher based function to parse complex composite nodes.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_DNNL_COMP_OP_MATCHER_H_
#define TVM_RELAY_BACKEND_CONTRIB_DNNL_COMP_OP_MATCHER_H_

#include <tvm/relay/function.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../ir/dataflow_matcher_impl.h"

/*!
 * \brief Converter value to dmlc attr acceptable format
 *
 * \tparam T type of value (auto deduction)
 * \param val value to convert
 * \return resulting dmlc object
 */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(std::vector<std::string>{std::to_string(val)});
  return dmlc::any{attr};
}

template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(std::vector<std::string>{val});
  return dmlc::any{attr};
}

template <typename T,
          std::enable_if_t<std::is_same<T, std::vector<std::string>>::value, bool> = true>
dmlc::any dmlc_attr(const T& val) {
  std::vector<dmlc::any> attr;
  attr.emplace_back(val);
  return dmlc::any{attr};
}

/*! \brief Constructor of const scalar expression with defined type */
tvm::relay::Expr constant(float val) {
  auto value = tvm::runtime::NDArray::Empty({}, tvm::DataType::Float(32), {kDLCPU, 0});
  value.CopyFromBytes(&val, sizeof(val));
  auto res = tvm::relay::Constant(value);
  tvm::relay::transform::InferTypeLocal(res);
  return res;
}

tvm::relay::DFPattern IsDnnlAct() {
  using namespace tvm::relay;
  auto dnnl_comp_act = IsWildcard().HasAttr({{"DnnlActType", tvm::String("gelu_erf")}});
  return IsOp("nn.relu") || IsOp("clip") || dnnl_comp_act;
}

/*!
 * \brief Simple helper to accumulate composite function arguments and corresponding attributes
 * with indexes of them.
 */
class ArgPacker {
 public:
  ArgPacker(std::unordered_map<std::string, dmlc::any>* attrs, std::vector<tvm::relay::Expr>* args)
      : attrs_(attrs), args_(args) {}

  int Put(const tvm::relay::Expr& arg, std::string tag_name = "") {
    if (!arg.defined()) return -1;
    int idx = args_->size();
    args_->push_back(arg);
    if (!tag_name.empty()) {
      (*attrs_)[tag_name] = dmlc_attr(idx);
    }
    return idx;
  }

  void PutActivation(const tvm::relay::Expr& act, const tvm::relay::Expr& act_scl) {
    auto act_cn = act.as<tvm::relay::CallNode>();
    ICHECK(act_cn);

    auto act_scl_val = act_scl.defined() ? act_scl : constant(1.0);  // 1.0 means no scaling

    std::string act_name = "";
    if (auto op = act_cn->op.as<tvm::relay::OpNode>())
      act_name = op->name;
    else if (auto fn = act_cn->op.as<tvm::relay::FunctionNode>())
      act_name = fn->GetAttr<tvm::String>("DnnlActType").value();
    ICHECK(!act_name.empty());

    // Default values. TODO: check if it acceptable for all activations
    auto act_alpha = constant(0.0);
    auto act_beta = constant(0.0);

    if (act_name == "clip")
      act_beta = constant(255.0);

    std::vector<std::string> clip_attr{act_name};
    clip_attr.push_back(std::to_string(Put(act_scl)));
    clip_attr.push_back(std::to_string(Put(act_alpha)));
    clip_attr.push_back(std::to_string(Put(act_beta)));
    (*attrs_)["activation"] = dmlc_attr(clip_attr);
  }

 private:
  std::unordered_map<std::string, dmlc::any>* attrs_;
  std::vector<tvm::relay::Expr>* args_;
};

const tvm::relay::CallNode* ParseQnnConvComp(const tvm::relay::FunctionNode& comp_fn,
                                             std::unordered_map<std::string, dmlc::any>* ext_attrs,
                                             std::vector<tvm::relay::Expr>* args) {
  using namespace tvm::relay;

  // Pattern
  auto src = IsWildcard();
  auto wgh = IsWildcard();
  auto sum_src = IsWildcard();
  auto bias = IsConstant();

  auto o_scl = IsConstant();
  auto act_scl = IsConstant();
  auto sum_scl = IsConstant();
  auto dst_zp = IsConstant();

  DFPattern cnv, act, pat;

  cnv = IsOp("qnn.conv2d")({src, wgh, IsConstant(), IsConstant(), IsConstant(), IsConstant()});
  pat = IsOp("cast")({cnv});
  pat = IsOp("add")({pat, bias}) || pat;
  pat = IsOp("multiply")({pat, o_scl});
  act = IsDnnlAct()({pat});
  pat = IsOp("multiply")({act, act_scl}) || act;
  pat = IsOp("add")({pat, sum_scl * IsOp("cast")({sum_src})}) || pat;
  pat = IsOp("add")({pat, dst_zp}) || pat;
  pat = IsOp("cast")({pat});

  // Check pattern match
  auto indexed_body = CreateIndexedGraph(comp_fn.body);
  DFPatternMatcher matcher(indexed_body.get());
  auto res = matcher.Match(pat, comp_fn.body);
  ICHECK(res) << "Mismatch of DNNL partitioner and codegen logic";

  // Handle arguments in deterministic order
  auto map = matcher.GetMemo();
  auto find = [&map](const DFPattern& pat) -> tvm::relay::Expr {
    if (map.count(pat)) return map.at(pat)[0];
    return {};
  };

  ArgPacker arg_holder(ext_attrs, args);
  arg_holder.Put(find(src));
  arg_holder.Put(find(wgh));
  arg_holder.Put(find(bias), "bias_idx");
  arg_holder.Put(find(sum_src), "sum_idx");
  arg_holder.Put(find(o_scl), "o_scl_idx");
  arg_holder.Put(find(act_scl), "act_scl_idx");
  arg_holder.Put(find(sum_scl), "sum_scl_idx");
  arg_holder.Put(find(dst_zp), "dst_zp_idx");
  arg_holder.PutActivation(find(act), find(act_scl));

  return map.at(cnv)[0].as<CallNode>();
}

const tvm::relay::CallNode* ParseQnnDenseComp(const tvm::relay::FunctionNode& comp_fn,
                                              std::unordered_map<std::string, dmlc::any>* ext_attrs,
                                              std::vector<tvm::relay::Expr>* args) {
  using namespace tvm::relay;

  // Pattern
  auto src = IsWildcard();
  auto wgh = IsWildcard();
  auto sum_src = IsWildcard();
  auto bias = IsConstant();

  auto o_scl = IsConstant();
  auto act_scl = IsConstant();
  auto sum_scl = IsConstant();
  auto dst_zp = IsConstant();

  DFPattern dns, act, pat;

  dns = IsOp("qnn.dense")({src, wgh, IsConstant(), IsConstant(), IsConstant(), IsConstant()});
  pat = IsOp("cast")({dns});
  pat = IsOp("add")({pat, bias}) || pat;
  pat = IsOp("multiply")({pat, o_scl});
  act = IsDnnlAct()({pat});
  pat = IsOp("multiply")({act, act_scl}) || act;
  pat = IsOp("add")({pat, sum_scl * IsOp("cast")({sum_src})}) || pat;
  pat = IsOp("add")({pat, dst_zp}) || pat;
  pat = IsOp("cast")({pat});

  // Check pattern match
  auto indexed_body = CreateIndexedGraph(comp_fn.body);
  DFPatternMatcher matcher(indexed_body.get());
  auto res = matcher.Match(pat, comp_fn.body);
  ICHECK(res) << "Mismatch of DNNL partitioner and codegen logic";

  // Handle arguments in deterministic order
  auto memo = matcher.GetMemo();
  auto find = [&memo](const DFPattern& pat) -> tvm::relay::Expr {
    if (memo.count(pat)) return memo.at(pat)[0];
    return {};
  };

  ArgPacker arg_holder(ext_attrs, args);
  arg_holder.Put(find(src));
  arg_holder.Put(find(wgh));
  arg_holder.Put(find(bias), "bias_idx");
  arg_holder.Put(find(sum_src), "sum_idx");
  arg_holder.Put(find(o_scl), "o_scl_idx");
  arg_holder.Put(find(act_scl), "act_scl_idx");
  arg_holder.Put(find(sum_scl), "sum_scl_idx");
  arg_holder.Put(find(dst_zp), "dst_zp_idx");
  arg_holder.PutActivation(find(act), find(act_scl));

  return memo.at(dns)[0].as<CallNode>();
}

/*!
 * Parse composite function and return real args, additional attributes and root call node
 * @param comp_fn composite function to parse
 * @param ext_attrs attr collection with additional attributes
 * @param args real arguments of node
 * @return root call node
 */
const tvm::relay::CallNode* ParseComposite(const tvm::relay::FunctionNode& comp_fn,
                                           std::unordered_map<std::string, dmlc::any>* ext_attrs,
                                           std::vector<tvm::relay::Expr>* args) {
  auto comp = comp_fn.GetAttr<tvm::String>(tvm::relay::attr::kComposite);
  ICHECK(comp.defined()) << "DNNL JSON runtime only supports composite functions.";
  auto name = comp.value();

  const tvm::relay::CallNode* res = nullptr;
  if (name == "dnnl.qnn.conv2d")
    res = ParseQnnConvComp(comp_fn, ext_attrs, args);
  else if (name == "dnnl.qnn.dense")
    res = ParseQnnDenseComp(comp_fn, ext_attrs, args);
  return res;
}

#endif  // TVM_RELAY_BACKEND_CONTRIB_DNNL_COMP_OP_MATCHER_H_
