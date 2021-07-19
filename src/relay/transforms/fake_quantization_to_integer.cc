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
 * \file src/relay/transforms/quantize_fake_quantization.cc
 * \brief A pass for taking fake quantized graphs and converting them
 * to actual integer operations.
 */

#include <tvm/ir/affine_type.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

/* Description of FakeQuantizationToInteger
 *
 * The purpose of this pass is to find regions of the graph that follow
 * the general pattern:
 *
 *   x    w
 *   |    |
 *   dq   dq
 *    \   /
 *     op1
 *      |
 *     op2
 *      |
 *      q
 *
 * and convert them into subgraphs with actual integer operations on x and w
 *
 * The pass does this via a multi-pass approach:
 *
 * The main pass is a MixedModeMutator that traverses the full graph searching for
 * quantize operations
 *
 * The second pass is an ExprVisitor that recursively searches for subgraphs leading to the
 * quantize for subtraphs bounded by dequantize operations. This pass extracts the affine
 * types of the inputs for later processing, where affine denotes the transformation
 * x_real = (x_affine - zero_point) * scale
 *
 * The third pass is an ExprMutator that recursively rewrites the subgraphs using packed funcs
 * registered with the FTVMFakeQuantizationToInteger attribute. These packed funcs rewrite
 * the ops based on the affine types of their inputs and then return the affine types of the
 * new rewriten ops to pass that information down the stack during rewrite.
 *
 * After the second and third passes run, the first pass replaces the quantize with the
 * rewritten subgraph and the processing continues
 */

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using ExprMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
using AffineTypeMap = Map<Expr, AffineType>;

using FTVMFakeQuantizationToInteger =
    runtime::TypedPackedFunc<Array<ObjectRef>(const Expr& expr, const AffineTypeMap& map)>;


// TODO(amalyshe) move to some header like where dense make is defined
namespace qnn {
Expr MakeDequantize(Expr data, Expr input_scale, Expr input_zero_point, int axis);
}  // namespace qnn


class PotentialQExtractor : public ExprVisitor {
public:
  const std::set<const CallNode *> GetLatestPotentialQuantized(const Expr &expr) {
    VisitExpr(expr);
    for (size_t i = orderVisits_.size(); i > 0; i--) {
      if (Downcast<Op>(orderVisits_[i - 1].as<CallNode>()->op) != dequantize_op_ &&
          Downcast<Op>(orderVisits_[i - 1].as<CallNode>()->op) != divide_op_ &&
          candidates_.find(orderVisits_[i - 1]) == candidates_.end() &&
          callers_.find(orderVisits_[i - 1]) != callers_.end()) {
        // going over all args and if all of them are potential quantized - mark
        // current as quantize and remove args from potential quantized
        bool callersq = nonquantizable_.find(orderVisits_[i - 1]) == nonquantizable_.end();
        for (auto c : callers_[orderVisits_[i - 1]]) {
          if (candidates_.find(c) == candidates_.end()) {
            callersq = false;
          }
        }
        if (callersq) {
          candidates_.insert(orderVisits_[i - 1]);
          for (auto c : callers_[orderVisits_[i - 1]]) {
            candidates_.erase(c);
          }
        }
      }
    }
    std::set<const CallNode *> nodes;
    for (auto c : candidates_) {
      nodes.insert(c.as<CallNode>());
    }

    return nodes;
  }

  void VisitExpr(const Expr &expr)override {
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        /*expr.as<ConstantNode>() == nullptr &&*/
        !stack_.empty()) {
      nonquantizable_.insert(stack_[stack_.size() - 1]);
      const OpNode *op_node = stack_[stack_.size() - 1].as<CallNode>()->op.as<OpNode>();
    }

    if (visiteuniq_.find(expr) == visiteuniq_.end()) {
      visiteuniq_.insert(expr);
      if (expr.as<CallNode>()) {
        orderVisits_.push_back(expr);
        if (!stack_.empty()) {
          auto it = callers_.find(stack_[stack_.size() - 1]);
          if (it != callers_.end()) {
            it->second.push_back(expr);
          } else {
            callers_[stack_[stack_.size() - 1]] = { expr };
          }
        }
        stack_.push_back(expr);
      }

      ExprVisitor::VisitExpr(expr);
      if (expr.as<CallNode>()) {
        stack_.pop_back();
      }
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

protected:
  void VisitExpr_(const CallNode *call_node)override {

    if (call_node->op == dequantize_op_) {
      candidates_.insert(stack_[stack_.size() - 1]);
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  // TODO(amalyshe) remove hardcoding of non quantized divide op
  const Op divide_op_ = Op::Get("divide");
  std::map<Expr, int> providers_;
  std::vector<Expr> stack_;
  std::map<Expr, std::vector<Expr>> callers_;
  std::set<Expr> candidates_;
  std::vector<Expr> orderVisits_;
  std::set<Expr> visiteuniq_;
  std::set<Expr> nonquantizable_;
};


class SubgraphExtractor : public ExprVisitor {
 public:
  const ExprSet GetSubgraph(const Expr& expr) {
    VisitExpr(expr);
    ExprSet subgraph;
    if (const CallNode *call_node = expr.as<CallNode>()) {
      if (call_node->op != dequantize_op_) {
        VisitExpr(expr);
        if (is_fake_quantized_) {
          for (auto kv : this->visit_counter_) {
            if (GetRef<ObjectRef>(kv.first).as<CallNode>() &&
                Downcast<Expr>(GetRef<ObjectRef>(kv.first)) != expr) {
              subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
            }
          }
        }
      }
    }
    return subgraph;
  }
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    if (call_node->op == quantize_op_) {
      // Only look at arg0 for quantize
      VisitExpr(call_node->args[0]);
      // Collect type of quantize ops
      affine_types_.Set(GetRef<Expr>(call_node),
                        TensorAffineType(call_node->args[1], call_node->args[2],
                                         call_node->checked_type().as<TensorTypeNode>()->dtype));
    } else if (call_node->op == dequantize_op_) {
      // Collect type of dequantize ops
      affine_types_.Set(
          GetRef<Expr>(call_node),
          TensorAffineType(call_node->args[1], call_node->args[2],
                           call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype));
    } else {
      // run normally on everything else.
      ExprVisitor::VisitExpr_(call_node);
    }
  }

  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
};

class SubgraphMutator : public ExprMutator {
 public:
  SubgraphMutator(ExprSet subgraph, AffineTypeMap affine_types)
      : subgraph_(subgraph), affine_types_(affine_types) {}

  Expr MutateSubgraph(const Expr& expr) {
    if (subgraph_.size() == 0) {
      return expr;
    }
    ICHECK(expr.as<CallNode>());
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    for (auto node : subgraph_) {
      if (!fqfq.count(Downcast<Op>(node.as<CallNode>()->op))) {
        // Only modify the subgraph if we have translation
        // rules for every op
        return expr;
      }
    }
    return Mutate(expr);
  }

 protected:
  Expr VisitExpr_(const CallNode* call_node) {
    Expr out;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    if (!call_node_) {
      call_node_ = call_node;
    }
    Op op = Downcast<Op>(call_node->op);
    if (fqfq.count(op)) {
      Expr expr;
      if (op == dequantize_op_) {
        expr = GetRef<Expr>(call_node);
      } else {
        expr = ExprMutator::VisitExpr_(call_node);
        // Set the current op to the output type, useful if we can't deduce output parameters
        // from input parameters
        affine_types_.Set(expr, out_type_);
      }
      // Call the rewrite
      Array<ObjectRef> vals = fqfq[op](expr, affine_types_);
      // Save teh outputs of the rewrite
      ICHECK(vals.size() == 2)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);
      out_type_ = Downcast<AffineType>(vals[1]);
      affine_types_.Set(out, out_type_);
      if (call_node_ == call_node &&
          call_node->op != quantize_op_) {
        const TensorAffineTypeNode *tatn = vals[1].as<TensorAffineTypeNode>();
        ICHECK(tatn);
        out = qnn::MakeDequantize(out, tatn->scale, tatn->zero_point, -1);
      }
    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  ExprSet subgraph_;
  AffineTypeMap affine_types_;
  AffineType out_type_;
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  // TODO(amalyshe) remove call_node_  and move adding of dequantize to the MutateSubgraph from VisitExpr_
  const CallNode *call_node_ = nullptr;
};

class FakeQuantizationRewriter : public MixedModeMutator {
 public:
  FakeQuantizationRewriter(std::set<const CallNode *> nodes) : nodes_(nodes) { }

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (nodes_.find(pre) != nodes_.end()) {
      SubgraphExtractor extractor;
      ExprSet subgraph = extractor.GetSubgraph(GetRef<Expr>(pre));
      AffineTypeMap affine_types = extractor.GetAffineTypes();

      ExprSet post_subgraph;
      AffineTypeMap post_affine_types;

      for (auto kv : affine_types) {
        if (pre == kv.first.as<CallNode>()) {
          // we havent memoized the current op yet
          post_affine_types.Set(post, kv.second);
        } else {
          post_affine_types.Set(memo_.at(kv.first), kv.second);
        }
      }
      for (auto expr : subgraph) {
        post_subgraph.insert(memo_[expr]);
      }
      Expr out = SubgraphMutator(post_subgraph, post_affine_types).MutateSubgraph(post);
      return out;
    }
    return post;
  }
  std::set<const CallNode *> nodes_;
};

Expr FakeQuantizationToInteger(const Expr& expr, const IRModule& mod) {
  PotentialQExtractor pqe;
  std::set<const CallNode *> graph = pqe.GetLatestPotentialQuantized(expr);
  return FakeQuantizationRewriter(graph).Mutate(expr);
}

namespace transform {

Pass FakeQuantizationToInteger() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FakeQuantizationToInteger(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "FakeQuantizationToInteger", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FakeQuantizationToInteger")
    .set_body_typed(FakeQuantizationToInteger);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
